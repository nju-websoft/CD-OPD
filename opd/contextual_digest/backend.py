from __future__ import annotations
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint
from .graph import GraphError


class HFBackend:
    def __init__(self, model, *, sample_bfloat16=True, score_chunk_tokens=128,
                 forward_parameter_dtype=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.sample_bfloat16 = sample_bfloat16
        self.score_chunk_tokens = score_chunk_tokens
        self.forward_parameter_dtype = forward_parameter_dtype


        self.external_checkpointing = bool(forward_parameter_dtype is not None
                                           and getattr(model, 'is_gradient_checkpointing', False))
        if self.external_checkpointing:


            model.gradient_checkpointing_disable()
        self._grammar_compiler = None
        self.clear_cache()

    def _forward_parameters(self):
        if self.forward_parameter_dtype is None:
            return None


        return {id(p): p.to(dtype=self.forward_parameter_dtype) for p in self.model.parameters()}

    def _forward(self, module, parameters, *args, **kwargs):
        if parameters is None:
            return module(*args, **kwargs)
        values = {name: parameters[id(p)] for name, p in module.named_parameters()}
        return torch.func.functional_call(module, values, args, kwargs, tie_weights=True)

    def clear_cache(self):
        self._prefix = []
        self._past = None
        self._logits = None

    @torch.no_grad()
    def next_logits(self, prefix):
        if not prefix:
            raise ValueError("An autoregressive prefix cannot be empty")
        if prefix == self._prefix:
            return self._logits
        if self._past is None or prefix[:len(self._prefix)] != self._prefix:
            self.clear_cache()
        suffix = prefix[len(self._prefix):]
        if not suffix:
            raise RuntimeError("Invalid KV cache prefix")
        self.model.eval()
        context = torch.autocast("cuda", dtype=torch.bfloat16) if (
            self.device.type == "cuda" and self.sample_bfloat16
        ) else nullcontext()
        with context:
            out = self._forward(self.model, self._forward_parameters(),
                input_ids=torch.tensor([suffix], device=self.device),
                past_key_values=self._past, use_cache=True, logits_to_keep=1, return_dict=True,
            )
        self._past = out.past_key_values
        self._prefix = list(prefix)
        self._logits = out.logits[0, -1].float()
        return self._logits

    def log_probs(self, prompt_ids, response_ids, *, grad=False):


        if not prompt_ids or not response_ids:
            raise ValueError("Scoring requires nonempty prompt and response")
        self.clear_cache()
        model = self.model
        if getattr(model.config, "final_logit_softcapping", None):
            raise ValueError("The reference scorer does not support logit softcapping")
        decoder = model.base_model
        head = model.get_output_embeddings()
        ids = torch.tensor([prompt_ids + response_ids], device=self.device)
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if (
            self.device.type == "cuda" and self.sample_bfloat16
        ) else nullcontext()
        with (torch.enable_grad() if grad else torch.no_grad()), precision:
            model.train(grad)
            parameters = self._forward_parameters()
            def decode(input_ids):
                return self._forward(decoder, parameters, input_ids=input_ids,
                                     use_cache=False, return_dict=True).last_hidden_state[0]
            hidden = checkpoint(decode, ids, use_reentrant=False) if grad and self.external_checkpointing else decode(ids)
            hidden = hidden[len(prompt_ids) - 1:-1]
            targets = ids[0, len(prompt_ids):]

            def project(states, tokens):
                logits = self._forward(head, parameters, states).float()
                return logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)

            values = []
            for start in range(0, len(response_ids), self.score_chunk_tokens):
                h = hidden[start:start + self.score_chunk_tokens]
                t = targets[start:start + self.score_chunk_tokens]
                values.append(checkpoint(project, h, t, use_reentrant=False) if grad else project(h, t))
            return torch.cat(values)

    @torch.no_grad()
    def generate_private(self, tokenizer, messages, max_tokens, max_model_len, schema):

        self.clear_cache()
        self.last_private_generation = None
        self.model.eval()
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
        if len(prompt) + max_tokens > max_model_len:
            raise GraphError("Complete graph input plus output budget exceeds teacher context; input was not truncated")
        ids = torch.tensor([prompt], device=self.device)
        import xgrammar as xgr
        from .constrained import GraphLogitsProcessor
        if self._grammar_compiler is None:
            info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=self.model.config.vocab_size,
                stop_token_ids=tokenizer.eos_token_id,
            )
            self._grammar_compiler = xgr.GrammarCompiler(info)


        with torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext():
            output = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id, use_cache=True,
                logits_processor=[GraphLogitsProcessor(self._grammar_compiler, schema, tokenizer, check_source_states=True)],
            )
        emitted = output[0, len(prompt):].tolist()
        text = tokenizer.decode(emitted, skip_special_tokens=True)
        self.last_private_generation = {
            "prompt_tokens": len(prompt), "generated_tokens": len(emitted),
            "ended_with_eos": bool(emitted and emitted[-1] == tokenizer.eos_token_id),
            "reached_token_limit": len(emitted) == max_tokens,
        }
        self.clear_cache()
        return text
