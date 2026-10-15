import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-size', choices=['0.6b', '1.7b', 'both'], default='both')
    parser.add_argument('--output-dir', type=Path, default=Path('models'))
    parser.add_argument('--sources', type=Path, default=Path('configs/model_sources.json'))
    args = parser.parse_args()
    sizes = ['0.6b', '1.7b'] if args.student_size == 'both' else [args.student_size]
    names = [f'Qwen3-{size[:-1]}B' for size in sizes] + ['Qwen3-4B-Instruct-2507']
    import json
    sources = json.loads(args.sources.read_text())
    for name in names:
        snapshot_download(repo_id=sources[name]['repo_id'], revision=sources[name]['revision'],
                          local_dir=args.output_dir / name)


if __name__ == '__main__':
    main()
