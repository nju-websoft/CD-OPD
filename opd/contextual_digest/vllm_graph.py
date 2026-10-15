import os

import torch
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .constrained import GraphLogitsProcessor, STRUCTURAL_PROTOCOLS


GRAPH_CACHE_LIMIT_BYTES = 512 * 1024 * 1024


class ParagraphStopProcessor:


    def __init__(self, rule, eos):
        self.tokens, self.count, self.eos = set(rule['token_ids']), rule['count'], eos

    def __call__(self, output_ids, logits):
        if sum(token in self.tokens for token in output_ids) >= self.count:
            logits.fill_(-torch.inf)
            logits[self.eos] = 0
        return logits


class VLLMGraphProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        import xgrammar as xgr
        path = os.environ.get('META_GRAPH_TOKENIZER_PATH', vllm_config.model_config.tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=vllm_config.model_config.get_vocab_size(),
            stop_token_ids=self.tokenizer.eos_token_id), cache_limit_bytes=GRAPH_CACHE_LIMIT_BYTES)

    def is_argmax_invariant(self):
        return False

    @classmethod
    def validate_params(cls, sampling_params):
        extra = sampling_params.extra_args or {}
        schema = extra.get('meta_graph_schema')
        if schema is not None and not isinstance(schema, dict):
            raise ValueError('meta_graph_schema must be a JSON schema object')
        rule = extra.get('relay_paragraph_stop')
        if rule is not None:
            if schema is not None:
                raise ValueError('Graph generation cannot use Relay paragraph stopping')
            if (not isinstance(rule, dict) or set(rule) != {'token_ids', 'count'}
                    or not isinstance(rule['count'], int) or rule['count'] < 1
                    or not isinstance(rule['token_ids'], list)
                    or any(not isinstance(t, int) or t < 0 for t in rule['token_ids'])):
                raise ValueError('Invalid Relay paragraph stopping rule')

    def new_req_logits_processor(self, params):
        extra = params.extra_args or {}
        if extra.get('relay_paragraph_stop') is not None:
            eos = self.tokenizer.eos_token_id
            return ParagraphStopProcessor(extra['relay_paragraph_stop'], eos[0] if isinstance(eos, list) else eos)
        schema = extra.get('meta_graph_schema')
        if schema is None:
            return None
        brief_state = schema.get('$comment') in STRUCTURAL_PROTOCOLS
        processor = GraphLogitsProcessor(self.compiler, schema, self.tokenizer, check_source_states=not brief_state)
        processor.prompt_length = 1

        def apply(output_ids, logits):
            processor(torch.tensor([[0, *output_ids]], dtype=torch.long), logits.unsqueeze(0))
            return logits

        return apply
