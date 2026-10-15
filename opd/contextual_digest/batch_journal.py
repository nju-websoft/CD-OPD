from dataclasses import asdict

import torch

from .graph import SummarySpan
from .rollout import Trajectory


class BatchJournal:
    def __init__(self, journal):
        self.journal = journal

    def path(self, kind, index):
        if kind not in {'handoff', 'independent'} or index < 0:
            raise ValueError('Invalid batch journal key')
        return self.journal.directory / f'batch_{kind}_{index:05d}.pt'

    def save(self, kind, index, traces, rewards=None):
        for trace in traces:
            trace.validate()
        if not traces or (rewards is not None and len(rewards) != len(traces)):
            raise ValueError('Incomplete batch cache')
        target = self.path(kind, index)
        if target.exists():
            raise FileExistsError('A completed microbatch cannot be overwritten')
        value = {'identity': self.journal.identity, 'traces': [asdict(t) for t in traces], 'rewards': rewards,
                 'generator_states': [g.get_state() for g in self.journal.generators],
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = target.with_suffix('.incomplete')
        torch.save(value, temporary)
        temporary.replace(target)

    def load(self, kind, index, question_ids):
        path = self.path(kind, index)
        if not path.exists():
            return None
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['identity'] != self.journal.identity or [t['question_id'] for t in value['traces']] != question_ids:
            raise ValueError('Microbatch belongs to another Student snapshot or question order')
        traces = [Trajectory(**{**t, 'summary_spans': [SummarySpan(**s) for s in t['summary_spans']]})
                  for t in value['traces']]
        for trace in traces:
            trace.validate()
        for generator, state in zip(self.journal.generators, value['generator_states'], strict=True):
            generator.set_state(state)
        torch.set_rng_state(value['torch_rng'])
        if value['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(value['cuda_rng'])
        return traces, value['rewards']

    def load_prefix(self, kind, index, question_ids):

        path = self.path(kind, index)
        if not path.exists():
            return None
        value = torch.load(path, map_location='cpu', weights_only=False)
        saved_ids = [t['question_id'] for t in value['traces']]
        if not saved_ids or len(saved_ids) > len(question_ids) or saved_ids != question_ids[:len(saved_ids)]:
            raise ValueError('Cached microbatch is not the expected question prefix')
        return self.load(kind, index, saved_ids)
