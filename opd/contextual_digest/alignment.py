from dataclasses import asdict


def baseline_comparison(config, execution):
    expected = {
        'handoff_batch_size': 128, 'seed': 42, 'epochs': 1, 'save_every': 5,
        'lr': 1e-6, 'weight_decay': 0.01, 'adam_betas': (0.9, 0.999),
        'clip_ratio': 0.2, 'max_grad_norm': 1.0,
    }
    rows = [dict(field=name, current=getattr(config, name), baseline=value,
                 matches=(tuple(getattr(config, name)) if name == 'adam_betas' else getattr(config, name)) == value)
            for name, value in expected.items()]
    relay = dict(trigger_topk=5, max_takeovers=2, paragraphs_per_takeover=3,
                 max_takeover_tokens=256, max_prompt_tokens=2048,
                 max_response_tokens=16384, temperature=1.0, top_p=1.0)
    rows.extend(dict(field='relay.' + name, current=getattr(config.relay, name),
                     baseline=value, matches=getattr(config.relay, name) == value)
                for name, value in relay.items())
    return rows


def require_baseline_recipe(config, execution):
    rows = baseline_comparison(config, execution)
    errors = [row['field'] for row in rows if not row['matches']]
    if errors:
        raise ValueError('Shared settings differ from the paper recipe: ' + ', '.join(errors))
    return rows
