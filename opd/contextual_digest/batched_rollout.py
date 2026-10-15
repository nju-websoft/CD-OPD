import torch
from .rollout import Trajectory


def random_seed(generator):
    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())


def independent_batch(student_client, tokenizer, questions, samples, max_tokens, generator):

    expanded = [q for q in questions for _ in range(samples)]
    outputs = student_client.generate([
        {'prompt_ids': q.prompt_ids, 'max_tokens': max_tokens, 'temperature': 1.0, 'seed': random_seed(generator)}
        for q in expanded])
    eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    result = []
    for question, output in zip(expanded, outputs, strict=True):
        ids = output['token_ids']
        if not ids:
            raise RuntimeError('Independent Student returned an empty answer')
        trace = Trajectory(question.id, list(question.prompt_ids), list(ids), ['student'] * len(ids),
                           stop_reason='eos' if ids[-1] in eos else 'response_limit')
        trace.validate()
        result.append(trace)
    return result
