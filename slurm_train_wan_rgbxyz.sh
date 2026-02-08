#!/bin/bash
#SBATCH -J lvp_joint_rgbxyz
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=512G
#SBATCH --partition=dgx-b200-old-driver
#SBATCH --ntasks-per-node=8
# #SBATCH --exclude=dgx004,dgx008,dgx011

set -euo pipefail

source ~/.bashrc
source /vast/projects/jgu32/lab/mutian/miniconda3/etc/profile.d/conda.sh
conda activate ei_world_model

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ======== EDIT THESE (or set env vars before sbatch) ========
DATA_ROOT="${DATA_ROOT:-/path/to/your/rgbxyz_dataset_root}"
# Can be relative to DATA_ROOT (recommended) or an absolute path.
METADATA_CSV="${METADATA_CSV:-metadata.csv}"
RUN_NAME="${RUN_NAME:-joint_rgbxyz_train}"

WANDB_MODE="${WANDB_MODE:-online}" # online|offline|disabled
WANDB_ENTITY="${WANDB_ENTITY:-mt3566-columbia-university}"
WANDB_PROJECT="${WANDB_PROJECT:-lvp}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_offline}"

# Common overrides (tune as needed)
BATCH_SIZE="${BATCH_SIZE:-1}"
CKPT_EVERY="${CKPT_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# ============================================================

export WANDB_MODE="${WANDB_MODE}"
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "${WANDB_DIR}"

srun ./train_wan_rgbxyz.sh "${DATA_ROOT}" "${METADATA_CSV}" "${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.project="${WANDB_PROJECT}" \
  experiment.num_nodes=1 \
  experiment.training.batch_size="${BATCH_SIZE}" \
  experiment.training.data.num_workers="${NUM_WORKERS}" \
  experiment.validation.val_every_n_step=100000000 \
  experiment.training.checkpointing.every_n_train_steps="${CKPT_EVERY}"
