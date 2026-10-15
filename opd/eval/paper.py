import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/evaluation.json'))
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    command = [sys.executable, '-m', 'opd.eval.math_benchmarks', '--model', args.model,
        '--data_dir', args.data_dir, '--out_dir', args.output_dir,
        '--benches', ','.join(config['samples']),
        '--bench_n_samples', ','.join(str(x) for x in config['samples'].values()),
        '--temperature', str(config['temperature']), '--top_p', str(config['top_p']),
        '--max_new', str(config['max_new_tokens']), '--seed', str(config['seed']),
        '--tp', str(args.tensor_parallel_size), '--gpu_mem', str(args.gpu_memory_utilization)]
    if config['disable_thinking']:
        command.append('--disable_thinking')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
