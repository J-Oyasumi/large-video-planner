#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <data_root> <metadata_csv> [run_name] [extra hydra overrides...]"
  echo ""
  echo "Example:"
  echo "  $0 /path/to/dataset metadata.csv rgbxyz \\"
  echo "    experiment.training.batch_size=1 experiment.num_nodes=1 \\"
  echo "    wandb.mode=disabled"
  exit 1
fi

DATA_ROOT="$1"
METADATA_CSV="$2"
RUN_NAME="${3:-rgb}"
EXTRA_ARGS=("${@:4}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/main.py" ]]; then
  PROJECT_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/large-video-planner/main.py" ]]; then
  PROJECT_ROOT="${SCRIPT_DIR}/large-video-planner"
else
  echo "ERROR: cannot locate project root from ${SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

python -m main \
  +name="${RUN_NAME}" \
  experiment=exp_video \
  algorithm=wan_i2v \
  dataset=rgb \
  dataset.data_root="${DATA_ROOT}" \
  dataset.metadata_path="${METADATA_CSV}" \
  algorithm.lang_guidance=0 \
  algorithm.hist_guidance=0 \
  experiment.validation.val_every_n_step=100000000 \
  "${EXTRA_ARGS[@]}"
