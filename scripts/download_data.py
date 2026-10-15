import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'
BENCHMARKS = {
    'aime24': ('aime-24.parquet', 30),
    'aime25': ('aime-2025.parquet', 30),
    'aime26': ('aime26.parquet', 30),
    'math500': ('math500.parquet', 500),
    'amc23': ('amc23.parquet', 40),
    'olympiadbench': ('olympiadbench.parquet', 674),
    'hmmt_feb_2026': ('hmmt_feb_2026.parquet', 33),
    'hmmt_nov_2025': ('hmmt_nov_2025.parquet', 30),
}


def prepare(name, rows):
    if name in {'aime24', 'amc23'}:
        year = '2024' if name == 'aime24' else '2023'
        rows = [r for r in rows if Path(urlparse(r['url']).path).parent.name.startswith(year + '_')]
    output = []
    for row in rows:
        if name == 'train':
            problem, answer = row['prompt'].strip(), row['solution']
        elif name == 'olympiadbench':
            if row['language'] != 'English' or row['modality'] != 'Text-only' or row['context']:
                raise ValueError('Unexpected OlympiadBench subset')
            if len(row['final_answer']) != 1:
                raise ValueError('Expected one OlympiadBench reference answer')
            problem, answer = row['question'], row['final_answer'][0]
        else:
            problem, answer = row['problem'], row['answer']
        if not isinstance(problem, str) or not problem.strip() or answer is None:
            raise ValueError('Missing problem statement or reference answer')
        if isinstance(answer, float) and answer.is_integer():
            answer = int(answer)
        output.append(dict(data_source='math_dapo' if name == 'train' else name,
            prompt=[dict(role='system', content=SYSTEM), dict(role='user', content=problem)],
            reward_model=dict(style='rule', ground_truth=str(answer))))
    expected = 14116 if name == 'train' else BENCHMARKS[name][1]
    if len(output) != expected or (name != 'train' and len({r['prompt'][1]['content'] for r in output}) != expected):
        raise ValueError(f'{name}: unexpected subset size or duplicate problem statements')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('configs/data_sources.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--cache-dir', type=Path, default=Path('runtime/downloads'))
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    names = ['train', *BENCHMARKS]
    targets = {name: args.output_dir / ('train.parquet' if name == 'train' else 'eval/' + BENCHMARKS[name][0]) for name in names}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError('Prepared data already exists; choose an empty output directory')
    manifest = {}
    for name in names:
        source = sources[name]
        raw = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
            revision=source['revision'], filename=source['file'], local_dir=args.cache_dir / name))
        rows = ([json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
                if raw.suffix == '.jsonl' else pq.read_table(raw).to_pylist())
        prepared = prepare(name, rows)
        target = targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(prepared), target, compression='zstd')
        with raw.open('rb') as stream:
            raw_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        with target.open('rb') as stream:
            prepared_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest[name] = dict(source=source, problems=len(prepared),
                             raw_sha256=raw_hash, prepared_sha256=prepared_hash)
        print(json.dumps(dict(dataset=name, problems=len(prepared), output=str(target))), flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
