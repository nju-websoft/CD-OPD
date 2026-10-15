import json
from types import SimpleNamespace

import pytest
import torch

from opd.contextual_digest.config import HandoffConfig, RelayConfig
from opd.contextual_digest.digest import (ContextualDigestBuilder, compile_digest,
    parse_digest_graph, validate_digest_trajectory)
from opd.contextual_digest.graph import GraphError, SummaryBudgetExceeded
from opd.contextual_digest.losses import relay_token_losses
from opd.contextual_digest.native_rollout import NativeRelayRollout
from opd.contextual_digest.rollout import RelayRollout, relay_trigger


class Tokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, ids, **kwargs):
        return ''.join(chr(x) for x in ids)
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


def node(key, part, quote, summary, requires=(), replaces=()):
    return dict(id=key, sources=[dict(part=part, quote=quote)],
                requires=list(requires), replaces=list(replaces), summary=summary)


def fixture():
    request = dict(question='Find the original positive number x.',
        prefix='Let u=x+2. Then u(u+1)=30.', guidance='Factor the equation to get u=5.',
        response_offset=10, remaining_tokens=1024)
    raw = dict(nodes=[
        node('N1', 'question', 'original positive number x', 'x is the original positive number.'),
        node('N2', 'prefix', 'u=x+2', 'The substitution is u=x+2.', ['N1']),
        node('N3', 'guidance', 'u=5', 'The teacher result is u=5.', ['N2']),
        node('N4', 'prefix', 'u(u+1)=30', 'An irrelevant disconnected node.'),
    ])
    return request, raw


def test_dependency_closed_digest_preserves_variable_meanings_and_final_guidance():
    request, raw = fixture()
    tok = Tokenizer()
    nodes, order = parse_digest_graph(raw, request)
    digest = compile_digest(nodes, order, tok, request, 1024)
    assert digest.selected_nodes == ('N1', 'N2', 'N3')
    assert [s.node_id for s in digest.spans] == ['N1', 'N2', 'current_guidance']
    assert 'u=x+2' in tok.decode(digest.token_ids)
    assert 'irrelevant' not in tok.decode(digest.token_ids)
    last = digest.spans[-1]
    assert list(digest.token_ids[last.start-10:last.end-10]) == tok.encode(request['guidance'])
    assert all(a.end == b.start for a, b in zip(digest.spans, digest.spans[1:]))


def test_guidance_only_when_no_contextual_ancestors_are_selected():
    request, raw = fixture()
    raw['nodes'][2]['requires'] = []
    nodes, order = parse_digest_graph(raw, request)
    digest = compile_digest(nodes, order, Tokenizer(), request, 1024)
    assert list(digest.token_ids) == Tokenizer().encode(request['guidance'])
    assert len(digest.spans) == 1


def test_teacher_correction_replaces_superseded_prefix_claim_without_editing_prefix():
    request, raw = fixture()
    request['prefix'] += ' The original number is u.'
    request['guidance'] += ' The original number is x, not u.'
    raw['nodes'].extend([
        node('N5', 'prefix', 'The original number is u.', 'Wrong: original number is u.'),
        node('N6', 'guidance', 'The original number is x, not u.', 'Original number is x.', ['N2'], ['N5'])])
    raw['nodes'][2]['requires'].append('N5')
    prefix = request['prefix']
    nodes, order = parse_digest_graph(raw, request)
    assert 'N5' not in nodes and 'N6' in nodes['N3'].requires
    digest = compile_digest(nodes, order, Tokenizer(), request, 1024)
    assert 'Wrong:' not in Tokenizer().decode(digest.token_ids)
    assert request['prefix'] == prefix


@pytest.mark.parametrize('damage', ['cycle', 'source', 'reference', 'duplicate', 'no_guidance'])
def test_invalid_dependency_graph_cannot_silently_become_a_guidance_only_digest(damage):
    request, raw = fixture()
    if damage == 'cycle': raw['nodes'][0]['requires'] = ['N3']
    elif damage == 'source': raw['nodes'][0]['sources'][0]['quote'] = 'invented source'
    elif damage == 'reference': raw['nodes'][2]['requires'] = ['N99']
    elif damage == 'duplicate': raw['nodes'][1]['id'] = 'N1'
    else: raw['nodes'] = raw['nodes'][:2]
    with pytest.raises(GraphError):
        parse_digest_graph(raw, request)


def test_budget_failure_preserves_complete_dependency_closure_by_rejecting_truncation():
    request, raw = fixture()
    nodes, order = parse_digest_graph(raw, request)
    with pytest.raises(SummaryBudgetExceeded):
        compile_digest(nodes, order, Tokenizer(), request, len(request['guidance']))


def test_trigger_rejects_any_student_reflection_token_and_ignores_random_draft():
    student = torch.tensor([5., 4., 1., 0.])
    teacher = torch.tensor([0., 0., 9., 0.])
    assert relay_trigger(student, teacher, 0, {1, 2}, 2) == (False, 2)
    assert relay_trigger(student, teacher, 2, {2}, 2) == (True, 2)


def test_clipped_surrogate_uses_advantage_frozen_at_sampling_snapshot():
    old = torch.tensor([-2., -1.], requires_grad=True)
    current = (old.detach() + .05).requires_grad_()
    teacher = torch.tensor([-1., -2.], requires_grad=True)
    loss = relay_token_losses(current, old, teacher)
    loss.sum().backward()
    expected = -(teacher.detach() - old.detach()) * torch.exp(current.detach() - old.detach())
    torch.testing.assert_close(current.grad, expected)
    assert old.grad is None and teacher.grad is None


class Native:
    def __init__(self, tok):
        self.tok, self.pending, self.adds = tok, [], 0
    def call(self, command, **kwargs):
        if command == 'relay_add':
            self.adds += 1
            self.pending.extend(kwargs['requests'])
            return dict(accepted=[r['key'] for r in kwargs['requests']], pending=len(self.pending))
        if command == 'relay_advance':
            result = [dict(key=r['key'], token_ids=self.tok.encode('sWait.'),
                sources=['student'] + ['teacher']*5, returned=True, guidance_start=1) for r in self.pending]
            self.pending = []
            return dict(finished=result, pending=0)
        if command == 'relay_finish': return {}
        raise AssertionError(command)


class Student:
    def generate(self, requests):
        return [dict(token_ids=[ord('x'), 0]) for r in requests]


def test_native_student_continues_after_last_takeover_and_digest_stays_inside_total_budget():
    tok = Tokenizer()
    cfg = RelayConfig(max_takeovers=2, max_takeover_tokens=8, max_response_tokens=100)
    builder = ContextualDigestBuilder(None, tok, HandoffConfig(dependency_graph=False, max_summary_tokens=16))
    native = Native(tok)
    rollout = NativeRelayRollout(native, tok, cfg, builder, student_client=Student())
    trace, = rollout.generate_many([SimpleNamespace(id='q', text='Find x.', prompt_ids=[1])],
                                  torch.Generator().manual_seed(42), None)
    assert native.adds == 2 and len(trace.handoffs) == 2
    assert trace.response_ids[-2:] == [ord('x'), 0]
    assert trace.sources[-2:] == ['student', 'student'] and trace.stop_reason == 'eos'
    assert len(trace.summary_spans) == 2
    validate_digest_trajectory(trace, 100, 16)


def test_native_low_remaining_budget_still_finishes_with_student_only_tokens():
    tok = Tokenizer()
    cfg = RelayConfig(max_takeovers=2, max_takeover_tokens=8, max_response_tokens=25)
    builder = ContextualDigestBuilder(None, tok, HandoffConfig(dependency_graph=False, max_summary_tokens=16))
    native = Native(tok)
    trace, = NativeRelayRollout(native, tok, cfg, builder, student_client=Student()).generate_many(
        [SimpleNamespace(id='q', text='Find x.', prompt_ids=[1])], torch.Generator(), None)
    assert native.adds == 1 and trace.response_ids[-1] == 0
    assert len(trace.response_ids) <= 25


def test_hf_student_resumes_after_the_maximum_number_of_takeovers():
    tok = Tokenizer()
    class Backend:
        def __init__(self, token):
            self.token = token
        def clear_cache(self):
            pass
        def next_logits(self, prefix):
            logits = torch.full((128,), -1000.)
            logits[self.token if len(prefix) < 9 else 0] = 0
            return logits
        def generate(self, requests):
            raw = dict(nodes=[node('N1', 'guidance', 'A', 'A')])
            return [dict(token_ids=tok.encode(json.dumps(raw))) for _ in requests]
    student, teacher = Backend(ord('s')), Backend(ord('A'))
    cfg = RelayConfig(trigger_topk=1, max_takeovers=2, max_takeover_tokens=1,
                      max_response_tokens=30, paragraphs_per_takeover=1)
    builder = ContextualDigestBuilder(teacher, tok, HandoffConfig(max_summary_tokens=4))
    trace = RelayRollout(student, teacher, tok, cfg, {ord('A')}, set(), builder).generate(
        'q', 'Find x.', [1], torch.Generator(), torch.Generator())
    assert len(trace.handoffs) == 2
    assert len(trace.summary_spans) == 2
    assert trace.response_ids[-1] == 0 and trace.sources[-1] == 'student'
    assert trace.stop_reason == 'eos'


def test_private_digest_json_grammar_accepts_real_tokenizer_output():
    import os
    path = os.environ.get('CD_OPD_TEST_TOKENIZER')
    if not path:
        pytest.skip('Set CD_OPD_TEST_TOKENIZER for the optional tokenizer integration check')
    from transformers import AutoTokenizer
    import xgrammar as xgr
    from opd.contextual_digest.constrained import compile_graph_schema
    from opd.contextual_digest.digest import digest_schema
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
    matcher = xgr.GrammarMatcher(compile_graph_schema(compiler, digest_schema()))
    request, raw = fixture()
    for token in tokenizer.encode(json.dumps(raw), add_special_tokens=False):
        assert matcher.accept_token(token)
    nodes, order = parse_digest_graph(raw, request)
    digest = compile_digest(nodes, order, tokenizer, request, 1024)
    assert tokenizer.decode(digest.token_ids).endswith(request['guidance'])
