#!/usr/bin/env bash
#SBATCH -J lvp_infer
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=512G
#SBATCH --partition=dgx-b200

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ei_world_model

DATA_ROOT="${1:-/vast/projects/jgu32/lab/mutian/FoundationStereo}"
METADATA_CSV="${2:-droid_processed/metadata_test_original.csv}"
RUN_NAME="${3:-infer_rgbxyz}"
CKPT_PATH="${4:-outputs_mutian/2026-02-10/00-44-34/checkpoints/latest.ckpt}"
OUTPUT_DIR="${5:-outputs/infer_rgbxyz}"

srun --ntasks-per-node=4 --gpus-per-task=1 \
  python -m main \
    +name="${RUN_NAME}" \
    experiment=exp_video \
    experiment.tasks=[validation] \
    experiment.num_nodes=1 \
    experiment.validation.limit_batch=null \
    experiment.validation.batch_size=2 \
    experiment.validation.data.num_workers=15 \
    experiment.strategy=auto \
    algorithm=wan_rgbxyz \
    algorithm.model.tuned_ckpt_path=${CKPT_PATH} \
    algorithm.logging.save_local=true \
    algorithm.logging.save_dir="${OUTPUT_DIR}" \
    algorithm.logging.video_type=grid \
    algorithm.hist_guidance=1.5 \
    algorithm.lang_guidance=2.5 \
    dataset=test_rgbxyz \
    dataset.data_root="${DATA_ROOT}" \
    dataset.metadata_path="${METADATA_CSV}" \
    wandb.mode=offline \


