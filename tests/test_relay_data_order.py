from copy import deepcopy

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from opd.contextual_digest.data import OnlineTrainingStream, TrainingQuestion, relay_handoff_order


@pytest.mark.parametrize('epochs', [1, 3])
def test_handoff_indices_match_baseline_loader_for_all_epochs_and_dropped_tail(epochs):
    questions = [TrainingQuestion(str(i), str(i), [i + 1], str(i), 'test_fixture') for i in range(141)]
    sampler = RandomSampler(questions, generator=torch.Generator().manual_seed(42))
    loader = StatefulDataLoader(questions, batch_size=16, drop_last=True, sampler=sampler,
                                num_workers=8, collate_fn=list)
    expected = [[q.id for q in batch] for _ in range(epochs) for batch in loader]
    stream = OnlineTrainingStream(questions, 16, 8, seed=42, epochs=epochs)
    actual = []
    for handoff, task in stream:
        actual.append([q.id for q in handoff])
        assert not {q.id for q in handoff} & {q.id for q in task}
    assert actual == expected


def test_new_sampler_resume_is_exact_and_old_sampler_state_cannot_silently_resume():
    questions = [TrainingQuestion(str(i), str(i), [i + 1], str(i), 'test_fixture') for i in range(35)]
    first = OnlineTrainingStream(questions, 8, 4, seed=42, epochs=2)
    next(first)
    state = first.state_dict()
    restored = OnlineTrainingStream(questions, 8, 4, seed=42, epochs=2)
    restored.load_state_dict(state)
    assert list(first) == list(restored)
    old = deepcopy(state)
    del old['handoff_sampler']
    with pytest.raises(ValueError, match='old Python handoff shuffle'):
        OnlineTrainingStream(questions, 8, 4).load_state_dict(old)


def test_cuda_safe_spawn_preserves_baseline_index_order(monkeypatch):


    relay_handoff_order.cache_clear()
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    expected = relay_handoff_order(79, 16, 42, 1)
    relay_handoff_order.cache_clear()
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    assert relay_handoff_order(79, 16, 42, 1) == expected
    relay_handoff_order.cache_clear()
