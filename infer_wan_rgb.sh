#!/bin/bash
#SBATCH -J lvp_infer_rgb
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --partition=dgx-b200

set -euo pipefail
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=False

eval "$(conda shell.bash hook)"
conda activate ei_world_model

DATA_ROOT="${1:-/vast/projects/jgu32/lab/mutian/FoundationStereo}"
METADATA_CSV="${2:-droid_processed/metadata_test_original.csv}"
RUN_NAME="${3:-infer_rgb}"
CKPT_PATH="${4:-data/ckpts/lvp_14B.ckpt}"
OUTPUT_DIR="${5:-outputs/infer_rgb}"

srun \
  python -m main \
    +name="${RUN_NAME}" \
    experiment=exp_video \
    experiment.tasks=[validation] \
    experiment.num_nodes=1 \
    experiment.validation.limit_batch=null \
    experiment.validation.batch_size=4 \
    experiment.validation.data.num_workers=15 \
    experiment.strategy=auto \
    algorithm=wan_i2v \
    algorithm.model.tuned_ckpt_path=${CKPT_PATH} \
    algorithm.logging.save_local=true \
    algorithm.logging.save_dir="${OUTPUT_DIR}" \
    algorithm.logging.video_type=grid \
    algorithm.hist_guidance=0.0 \
    algorithm.lang_guidance=0.0 \
    dataset=test_rgb \
    dataset.data_root="${DATA_ROOT}" \
    dataset.metadata_path="${METADATA_CSV}" \
    wandb.mode=offline


