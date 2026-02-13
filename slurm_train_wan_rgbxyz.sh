#!/bin/bash
#SBATCH -J lvp_joint_2e-5
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=1024G
#SBATCH --partition=dgx-b200
#SBATCH --ntasks-per-node=8
# Tip:
#   You can override GPU count at submit time:
#   - 2 GPUs: sbatch --gres=gpu:2 --ntasks-per-node=2 slurm_train_wan_rgbxyz.sh
#   - 4 GPUs: sbatch --gres=gpu:4 --ntasks-per-node=4 slurm_train_wan_rgbxyz.sh
#   - 8 GPUs: sbatch --gres=gpu:8 --ntasks-per-node=8 slurm_train_wan_rgbxyz.sh


set -euo pipefail

source ~/.bashrc
source /vast/projects/jgu32/lab/mutian/miniconda3/etc/profile.d/conda.sh
conda activate ei_world_model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/train_wan_rgbxyz.sh" ]]; then
  TRAIN_LAUNCHER="${REPO_ROOT}/train_wan_rgbxyz.sh"
elif [[ -x "${REPO_ROOT}/../train_wan_rgbxyz.sh" ]]; then
  TRAIN_LAUNCHER="${REPO_ROOT}/../train_wan_rgbxyz.sh"
else
  echo "ERROR: cannot find train_wan_rgbxyz.sh from REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi

detect_gpus_per_node() {
  local gpus_raw="${SLURM_GPUS_ON_NODE:-}"
  local ntasks_raw="${SLURM_NTASKS_PER_NODE:-}"

  if [[ -n "${gpus_raw}" ]]; then
    if [[ "${gpus_raw}" =~ ^[0-9]+$ ]]; then
      echo "${gpus_raw}"
      return
    fi
    if [[ "${gpus_raw}" =~ gpu:([0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
    if [[ "${gpus_raw}" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
    if [[ "${gpus_raw}" == *,* ]]; then
      awk -F',' '{print NF}' <<< "${gpus_raw}"
      return
    fi
  fi

  if [[ -n "${ntasks_raw}" ]]; then
    echo "${ntasks_raw%%(*}"
    return
  fi

  echo "1"
}

to_hydra_bool() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|y|on)
      echo "true"
      ;;
    0|false|no|n|off)
      echo "false"
      ;;
    *)
      echo "ERROR: invalid boolean value '${value}' (use 0/1/true/false)" >&2
      exit 1
      ;;
  esac
}

# ======== EDIT THESE (or set env vars before sbatch) ========
DATA_ROOT="/vast/projects/jgu32/lab/mutian/FoundationStereo"
# Can be relative to DATA_ROOT (recommended) or an absolute path.
METADATA_CSV="droid_processed/combined.csv"
RUN_NAME="${RUN_NAME:-joint_rgbxyz_$(date +%m%d)_49f_480x832_bs1x8}"

WANDB_MODE="${WANDB_MODE:-online}" # online|offline|disabled
WANDB_ENTITY="${WANDB_ENTITY:-mt3566-columbia-university}"
WANDB_PROJECT="${WANDB_PROJECT:-lvp}"
# W&B always writes local run files here (online/offline both use it).
WANDB_DIR="${WANDB_DIR:-${SLURM_SUBMIT_DIR:-${HOME}}/wandb}"

# Common overrides (tune as needed)
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CKPT_EVERY="${CKPT_EVERY:-500}"
VAL_EVERY="${VAL_EVERY:-5}"
VAL_SPLIT="${VAL_SPLIT:-all}"
VAL_SHUFFLE="${VAL_SHUFFLE:-1}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
if ! [[ "${CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || (( CPUS_PER_TASK < 1 )); then
  CPUS_PER_TASK=1
fi
DEFAULT_NUM_WORKERS=$(( CPUS_PER_TASK > 1 ? CPUS_PER_TASK - 1 : 1 ))
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || (( NUM_WORKERS < 1 )); then
  echo "ERROR: NUM_WORKERS must be a positive integer, got '${NUM_WORKERS}'" >&2
  exit 1
fi
if ! [[ "${GRAD_ACCUM}" =~ ^[0-9]+$ ]] || (( GRAD_ACCUM < 1 )); then
  echo "ERROR: GRAD_ACCUM must be a positive integer, got '${GRAD_ACCUM}'" >&2
  exit 1
fi
if (( NUM_WORKERS > CPUS_PER_TASK )); then
  echo "WARN: NUM_WORKERS=${NUM_WORKERS} > SLURM_CPUS_PER_TASK=${CPUS_PER_TASK}; capping to ${CPUS_PER_TASK}"
  NUM_WORKERS="${CPUS_PER_TASK}"
fi
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
LR="${LR:-5e-5}"
LR_SCHEDULER_NAME="${LR_SCHEDULER_NAME:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
TUNED_CKPT="${TUNED_CKPT:-data/ckpts/lvp_14B.ckpt}"
LOAD_CKPT="${LOAD_CKPT:-/vast/projects/jgu32/lab/mutian/large-video-planner-xyz/large-video-planner/outputs/2026-02-12/15-10-35/checkpoints/latest.ckpt}"
# weights: treat LOAD_CKPT as model-weights-only checkpoint (recommended with save_weights_only=True)
# resume:  pass LOAD_CKPT to Lightning ckpt_path and restore optimizer/scheduler/global step
LOAD_MODE="${LOAD_MODE:-weights}" # weights|resume
MODALITY_EMBED_SCALE="${MODALITY_EMBED_SCALE:-2.0}"
MODALITY_EMBED_NONZERO_INIT="${MODALITY_EMBED_NONZERO_INIT:-1}"
MODALITY_EMBED_INIT_STD="${MODALITY_EMBED_INIT_STD:-0.02}"
MODALITY_EMBED_REINIT_ON_START="${MODALITY_EMBED_REINIT_ON_START:-0}"
NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(detect_gpus_per_node)}"
# ============================================================

MODALITY_EMBED_NONZERO_INIT="$(to_hydra_bool "${MODALITY_EMBED_NONZERO_INIT}")"
MODALITY_EMBED_REINIT_ON_START="$(to_hydra_bool "${MODALITY_EMBED_REINIT_ON_START}")"
VAL_SHUFFLE="$(to_hydra_bool "${VAL_SHUFFLE}")"

if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || (( NUM_NODES < 1 )); then
  echo "ERROR: NUM_NODES must be a positive integer, got '${NUM_NODES}'" >&2
  exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE < 1 )); then
  echo "ERROR: GPUS_PER_NODE must be a positive integer, got '${GPUS_PER_NODE}'" >&2
  exit 1
fi
TOTAL_TASKS=$((NUM_NODES * GPUS_PER_NODE))

LOAD_ARGS=()
case "${LOAD_MODE,,}" in
  weights)
    if [[ -n "${LOAD_CKPT}" ]]; then
      TUNED_CKPT="${LOAD_CKPT}"
    fi
    ;;
  resume)
    if [[ -n "${LOAD_CKPT}" ]]; then
      LOAD_ARGS=(load="${LOAD_CKPT}")
    fi
    ;;
  none)
    ;;
  *)
    echo "ERROR: LOAD_MODE must be one of weights|resume|none, got '${LOAD_MODE}'" >&2
    exit 1
    ;;
esac

TUNED_CKPT_ARGS=()
if [[ -n "${TUNED_CKPT}" ]]; then
  TUNED_CKPT_ARGS=(algorithm.model.tuned_ckpt_path="${TUNED_CKPT}")
fi

export WANDB_MODE="${WANDB_MODE}"
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${WANDB_DIR}"

echo "[launch] num_nodes=${NUM_NODES}, gpus_per_node=${GPUS_PER_NODE}, total_tasks=${TOTAL_TASKS}"
echo "[launch] cpus_per_task=${CPUS_PER_TASK}, train_num_workers=${NUM_WORKERS}, val_num_workers=${VAL_NUM_WORKERS}"
echo "[launch] batch_size=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}"
echo "[launch] validation: split=${VAL_SPLIT}, shuffle=${VAL_SHUFFLE}, every_n_step=${VAL_EVERY}"
echo "[launch] load_mode=${LOAD_MODE}, load_ckpt=${LOAD_CKPT:-<none>}, tuned_ckpt=${TUNED_CKPT:-<none>}"
echo "[launch] modality_embed: scale=${MODALITY_EMBED_SCALE}, nonzero_init=${MODALITY_EMBED_NONZERO_INIT}, init_std=${MODALITY_EMBED_INIT_STD}, reinit_on_start=${MODALITY_EMBED_REINIT_ON_START}"

srun --ntasks="${TOTAL_TASKS}" --ntasks-per-node="${GPUS_PER_NODE}" \
  "${TRAIN_LAUNCHER}" "${DATA_ROOT}" "${METADATA_CSV}" "${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.project="${WANDB_PROJECT}" \
  experiment.num_nodes="${NUM_NODES}" \
  experiment.training.lr="${LR}" \
  experiment.training.batch_size="${BATCH_SIZE}" \
  experiment.training.optim.accumulate_grad_batches="${GRAD_ACCUM}" \
  experiment.training.data.num_workers="${NUM_WORKERS}" \
  experiment.validation.split="${VAL_SPLIT}" \
  experiment.validation.data.shuffle="${VAL_SHUFFLE}" \
  experiment.validation.data.num_workers="${VAL_NUM_WORKERS}" \
  algorithm.lr_scheduler.name="${LR_SCHEDULER_NAME}" \
  algorithm.lr_scheduler.num_warmup_steps="${LR_WARMUP_STEPS}" \
  algorithm.modality_embedding.scale="${MODALITY_EMBED_SCALE}" \
  algorithm.modality_embedding.nonzero_init="${MODALITY_EMBED_NONZERO_INIT}" \
  algorithm.modality_embedding.init_std="${MODALITY_EMBED_INIT_STD}" \
  algorithm.modality_embedding.reinit_on_start="${MODALITY_EMBED_REINIT_ON_START}" \
  experiment.validation.val_every_n_step="${VAL_EVERY}" \
  experiment.training.checkpointing.every_n_train_steps="${CKPT_EVERY}" \
  "${LOAD_ARGS[@]}" \
  "${TUNED_CKPT_ARGS[@]}"
