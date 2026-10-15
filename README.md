# Contextual Digest for On-Policy Distillation

Code for **CD-OPD**, combining Contextual Digest construction with online meta-reweighting.

![CD-OPD framework](Figure3.pdf)

## Installation

Linux, Python 3.12, and NVIDIA GPUs are required. The default configuration uses four GPUs: two student replicas and two frozen teacher replicas. Dependencies are listed in [requirements.txt](requirements.txt).

Run all commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Models and Data

Students: **Qwen3-0.6B** and **Qwen3-1.7B** in non-thinking mode. Teacher: **Qwen3-4B-Instruct-2507**.

Download the models, the English subset of DAPO-Math-17K, and the eight evaluation datasets:

```bash
python scripts/download_models.py --student-size both
python scripts/download_data.py --output-dir data --cache-dir runtime/downloads
```

Models are saved under `models/`; training data is written to `data/train.parquet` and evaluation data to `data/eval/`. Source revisions are recorded in [model_sources.json](configs/model_sources.json) and [data_sources.json](configs/data_sources.json).

## Training

Configurations: [Qwen3-0.6B](configs/qwen3_0.6b.json), [Qwen3-1.7B](configs/qwen3_1.7b.json), and [native execution](configs/execution_vllm.json).

Check a configuration:

```bash
python -m opd.contextual_digest.train_native \
  --config configs/qwen3_0.6b.json \
  --execution-config configs/execution_vllm.json --check-config
```

Train either student:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/qwen3_0.6b.json
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/qwen3_1.7b.json
```

For the HF backend, use `scripts/train_hf.sh` with the same configuration.

Checkpoints are saved every five updates under the configured `output_dir`, together with metrics and trajectory audits. Resume with the same backend and configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/qwen3_0.6b.json \
  --resume runs/cd_opd_0.6b/checkpoints/step_5
```

For a new run, set an unused `output_dir`.

## Evaluation

Evaluate a student checkpoint on all eight benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opd.eval.paper \
  --model runs/cd_opd_0.6b/checkpoints/step_80/student \
  --data-dir data/eval --output-dir eval_results/cd_opd_0.6b_step80 \
  --tensor-parallel-size 1
```

[evaluation.json](configs/evaluation.json) sets temperature 1.0, top-p 1.0, and a 32,768-token response limit. Evaluation uses student-only generation, with four responses per problem for MATH500 and OlympiadBench and 32 for the other benchmarks. The `avg@k` output corresponds to mean@k; the overall score is the unweighted mean across the eight benchmarks.

Compute Collaboration Failure Rate (CFR) from paired student-only, teacher-only, and takeover records:

```bash
python -m opd.eval.cfr \
  --student student_only.jsonl --teacher teacher_only.jsonl --takeover takeover.jsonl
```

## Tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q tests
```

