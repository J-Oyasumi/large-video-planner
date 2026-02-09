#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <data_root> <metadata_csv> [run_name] [extra hydra overrides...]"
  echo ""
  echo "Example:"
  echo "  $0 /path/to/dataset metadata.csv joint_rgbxyz \\"
  echo "    experiment.training.batch_size=1 experiment.num_nodes=1 \\"
  echo "    wandb.mode=disabled"
  exit 1
fi

DATA_ROOT="$1"
METADATA_CSV="$2"
RUN_NAME="${3:-joint_rgbxyz}"
EXTRA_ARGS=("${@:4}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/large-video-planner"

python -m main \
  +name="${RUN_NAME}" \
  experiment=exp_video \
  algorithm=wan_rgbxyz \
  dataset=rgbxyz \
  dataset.data_root="${DATA_ROOT}" \
  dataset.metadata_path="${METADATA_CSV}" \
  algorithm.lang_guidance=0 \
  algorithm.hist_guidance=0 \
  experiment.validation.val_every_n_step=100000000 \
  "${EXTRA_ARGS[@]}"

