#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <data_root> <metadata_csv> [row_index] [run_name] [ckpt_path] [output_dir] [extra hydra overrides...]"
  echo ""
  echo "Example:"
  echo "  $0 /vast/projects/jgu32/lab/mutian/FoundationStereo droid_processed/metadata_test_original.csv 0 \\"
  echo "     infer_rgbxyz_row0 outputs_mutian/2026-02-10/00-44-34/checkpoints/latest.ckpt outputs/infer_rgbxyz_row0"
  exit 1
fi

DATA_ROOT="$1"
METADATA_CSV="$2"
ROW_INDEX="${3:-0}"
RUN_NAME="${4:-infer_rgbxyz_row0}"
CKPT_PATH="${5:-outputs_mutian/2026-02-10/00-44-34/checkpoints/latest.ckpt}"
OUTPUT_DIR="${6:-outputs/infer_rgbxyz_row0}"
EXTRA_ARGS=()
if (( $# > 6 )); then
  EXTRA_ARGS=("${@:7}")
fi

if ! [[ "${ROW_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: row_index must be a non-negative integer, got '${ROW_INDEX}'" >&2
  exit 1
fi

if [[ "${METADATA_CSV}" = /* ]]; then
  METADATA_ABS="${METADATA_CSV}"
else
  METADATA_ABS="${DATA_ROOT%/}/${METADATA_CSV}"
fi

if [[ ! -f "${METADATA_ABS}" ]]; then
  echo "ERROR: metadata file not found: ${METADATA_ABS}" >&2
  exit 1
fi

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

TMP_METADATA_CSV="$(mktemp "${TMPDIR:-/tmp}/infer_one_sample.XXXXXX.csv")"
trap 'rm -f "${TMP_METADATA_CSV}"' EXIT

TARGET_LINE=$((ROW_INDEX + 2))
awk -v target="${TARGET_LINE}" 'NR==1 || NR==target' "${METADATA_ABS}" > "${TMP_METADATA_CSV}"

TMP_LINES="$(wc -l < "${TMP_METADATA_CSV}")"
if (( TMP_LINES < 2 )); then
  TOTAL_ROWS=$(( $(wc -l < "${METADATA_ABS}") - 1 ))
  echo "ERROR: row_index ${ROW_INDEX} is out of range. metadata rows: ${TOTAL_ROWS}" >&2
  exit 1
fi

echo "[single-sample infer] data_root=${DATA_ROOT}"
echo "[single-sample infer] metadata=${METADATA_ABS}"
echo "[single-sample infer] row_index=${ROW_INDEX}"
echo "[single-sample infer] run_name=${RUN_NAME}"
echo "[single-sample infer] ckpt_path=${CKPT_PATH}"
echo "[single-sample infer] output_dir=${OUTPUT_DIR}"
echo "[single-sample infer] temp_metadata=${TMP_METADATA_CSV}"

# Default to a single visible GPU for one-sample inference.
# Users can override by setting CUDA_VISIBLE_DEVICES before launching the script.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
echo "[single-sample infer] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# In interactive SLURM shells, stale distributed env vars can force MPI/DDP init.
# For one-sample local inference, clear them to guarantee single-process execution.
DIST_ENV_VARS=(
  RANK WORLD_SIZE LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT
  I_MPI_PMI_LIBRARY I_MPI_HYDRA_BOOTSTRAP I_MPI_HYDRA_BOOTSTRAP_EXEC
)
for _var in "${DIST_ENV_VARS[@]}"; do
  unset "${_var}" || true
done
while IFS='=' read -r _var _; do
  case "${_var}" in
    SLURM_*|PMI_*|PMIX_*|OMPI_*|I_MPI_*|MPICH_*|MPI_*)
      unset "${_var}" || true
      ;;
    HYDRA_*)
      if [[ "${_var}" != "HYDRA_FULL_ERROR" ]]; then
        unset "${_var}" || true
      fi
      ;;
  esac
done < <(env)
export PL_TORCH_DISTRIBUTED_BACKEND=nccl

python -m main \
  +_on_compute_node=True \
  +cluster.is_compute_node_offline=false \
  +name="${RUN_NAME}" \
  experiment=exp_video \
  'experiment.tasks=[validation]' \
  experiment.num_nodes=1 \
  experiment.validation.limit_batch=1 \
  experiment.validation.batch_size=1 \
  experiment.validation.data.num_workers=4 \
  experiment.strategy=auto \
  algorithm=wan_rgbxyz \
  algorithm.model.tuned_ckpt_path="${CKPT_PATH}" \
  algorithm.logging.save_local=true \
  algorithm.logging.save_dir="${OUTPUT_DIR}" \
  algorithm.logging.video_type=grid \
  algorithm.hist_guidance=1.5 \
  algorithm.lang_guidance=4.5 \
  dataset=test_rgbxyz \
  dataset.data_root="${DATA_ROOT}" \
  dataset.metadata_path="${TMP_METADATA_CSV}" \
  wandb.mode=offline \
  "${EXTRA_ARGS[@]}"
