#!/bin/bash
#SBATCH -J lvp_rgb
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --mem=256G
#SBATCH --partition=dgx-b200


set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ei_world_model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
cd "${REPO_ROOT}"

TRAIN_LAUNCHER="${REPO_ROOT}/train_wan_rgb.sh"

# ======== EDIT THESE (or set env vars before sbatch) ========
DATA_ROOT="/vast/projects/jgu32/lab/mutian/FoundationStereo"
# Can be relative to DATA_ROOT (recommended) or an absolute path.
METADATA_CSV="droid_processed/metadata_cap_norm_withcaption.csv"
RUN_NAME="${RUN_NAME:-only_rgb_$(date +%m%d)_49f_480x832_bs1x4}"

WANDB_MODE="${WANDB_MODE:-online}" # online|offline|disabled
WANDB_ENTITY="${WANDB_ENTITY:-hanjiang-nju}"
WANDB_PROJECT="${WANDB_PROJECT:-lvp}"
# W&B always writes local run files here (online/offline both use it).
WANDB_DIR="${WANDB_DIR:-${SLURM_SUBMIT_DIR:-${HOME}}/wandb}"

# Common overrides (tune as needed)
BATCH_SIZE="${BATCH_SIZE:-2}"
CKPT_EVERY="${CKPT_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-6}"
TUNED_CKPT="${TUNED_CKPT:-data/ckpts/lvp_14B.ckpt}"
# ============================================================

TUNED_CKPT_ARGS=()
if [[ -n "${TUNED_CKPT}" ]]; then
  TUNED_CKPT_ARGS=(algorithm.model.tuned_ckpt_path="${TUNED_CKPT}")
fi

export WANDB_MODE="${WANDB_MODE}"
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "${WANDB_DIR}"

srun "${TRAIN_LAUNCHER}" "${DATA_ROOT}" "${METADATA_CSV}" "${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.project="${WANDB_PROJECT}" \
  experiment.num_nodes=1 \
  experiment.training.batch_size="${BATCH_SIZE}" \
  experiment.training.data.num_workers="${NUM_WORKERS}" \
  experiment.validation.val_every_n_step=10 \
  experiment.training.checkpointing.every_n_train_steps="${CKPT_EVERY}" \
  "${TUNED_CKPT_ARGS[@]}"
