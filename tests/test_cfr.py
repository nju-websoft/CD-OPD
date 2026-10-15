import pytest
from opd.eval.cfr import collaboration_failure_rate


def rows(scores):
    return [dict(question_id='q', sample_id=i, correct=score) for i, score in enumerate(scores)]


def test_cfr_uses_only_pairs_where_both_independent_models_are_correct():
    report = collaboration_failure_rate(rows([1, 1, 0, 1]), rows([1, 1, 1, 0]), rows([0, 1, 0, 0]))
    assert report == dict(paired_responses=4, both_correct=2, failures=1, cfr=.5)


def test_cfr_is_undefined_without_a_both_correct_pair():
    assert collaboration_failure_rate(rows([0]), rows([1]), rows([0]))['cfr'] is None


def test_cfr_rejects_missing_or_duplicate_pairs():
    with pytest.raises(ValueError, match='identical'):
        collaboration_failure_rate(rows([1, 1]), rows([1]), rows([0]))
    with pytest.raises(ValueError, match='Duplicate'):
        collaboration_failure_rate(rows([1])*2, rows([1]), rows([0]))
