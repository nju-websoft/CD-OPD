from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from opd.contextual_digest.backend import HFBackend


def make_model(tied):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=71, hidden_size=48, intermediate_size=96,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=12, attention_dropout=0., tie_word_embeddings=tied)
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config).float().eval()


def bf16_parameter_reference(model):
    result = deepcopy(model)


    for parameter in result.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    return result


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_forward_values_match_bf16_model_and_master_gradients_survive_checkpointing(tied, checkpointed):
    master = make_model(tied)
    reference = bf16_parameter_reference(master)
    if checkpointed:
        master.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    backend = HFBackend(master, sample_bfloat16=False, score_chunk_tokens=128,
                        forward_parameter_dtype=torch.bfloat16)
    expected_backend = HFBackend(reference, sample_bfloat16=False, score_chunk_tokens=128)
    before = {name: p.detach().clone() for name, p in master.named_parameters()}
    versions = [p._version for p in master.parameters()]
    prompt, response = [1, 2, 3], [4, 5, 6, 7, 8, 9]
    actual = backend.log_probs(prompt, response, grad=True)
    expected = expected_backend.log_probs(prompt, response, grad=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.mean(), tuple(master.parameters()), retain_graph=True)
    b = torch.autograd.grad(expected.mean(), tuple(reference.parameters()))
    for actual_grad, expected_grad in zip(a, b):
        torch.testing.assert_close(actual_grad, expected_grad.float(), rtol=0, atol=0)
        assert actual_grad.dtype == torch.float32 and torch.isfinite(actual_grad).all()


    c = torch.autograd.grad(actual[:2].mean(), tuple(master.parameters()))
    assert all(torch.isfinite(g).all() for g in c)
    new_loss = -backend.log_probs(prompt, response, grad=True).mean()
    new_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in master.parameters())
    assert versions == [p._version for p in master.parameters()]
    assert all(torch.equal(p.detach(), before[name]) and p.dtype == torch.float32
               for name, p in master.named_parameters())
    assert backend.external_checkpointing == checkpointed


def test_parameter_cast_recomputes_after_master_update_instead_of_using_stale_views():
    model = make_model(True)
    backend = HFBackend(model, sample_bfloat16=False, forward_parameter_dtype=torch.bfloat16)
    prompt, response = [1, 2], [3, 4]
    before = backend.log_probs(prompt, response).clone()
    with torch.no_grad():
        model.get_input_embeddings().weight[3].add_(.25)
    after = backend.log_probs(prompt, response)
    reference = HFBackend(bf16_parameter_reference(model), sample_bfloat16=False)
    torch.testing.assert_close(after, reference.log_probs(prompt, response), rtol=0, atol=0)
    assert not torch.equal(before, after)
