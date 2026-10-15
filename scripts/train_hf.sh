#!/usr/bin/env bash
set -euo pipefail
CD_OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CD_OPD_ROOT}"
export PYTHONPATH="${CD_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CD_OPD_CONFIG="${1:-configs/qwen3_0.6b.json}"
if (( $# )); then shift; fi
CD_OPD_NPROC="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["student_devices"]))' "${CD_OPD_CONFIG}")"
exec python3 -m torch.distributed.run --standalone --nproc_per_node="${CD_OPD_NPROC}" \
    --module opd.contextual_digest.train --config "${CD_OPD_CONFIG}" "$@"
