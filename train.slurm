#!/bin/bash
#SBATCH --job-name=clip_flickr_train
#SBATCH --partition=gpu
#SBATCH --time=07:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/ramanagarajayaram.k/logs/%j_train.out
#SBATCH --error=/scratch/ramanagarajayaram.k/logs/%j_train.err

# --- Environment -------------------------------------------------------
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate vlm
module load cuda/12.3.0

# Avoids CUDA memory fragmentation on long-running training jobs by
# letting the allocator grow/shrink cached segments instead of only ever
# growing them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Redirect HuggingFace's cache to scratch space (must match data_flickr.py)
# so dataset/image downloads don't blow through the home directory quota.
export HF_HOME=/scratch/ramanagarajayaram.k/hf_cache
export HF_DATASETS_CACHE=/scratch/ramanagarajayaram.k/hf_cache

# --- Paths ---------------------------------------------------------------
REPO_DIR=/home/ramanagarajayaram.k/Vision-Language-Model
CHECKPOINT_DIR=/home/ramanagarajayaram.k/checkpoints/clip_flickr_v2

mkdir -p "$CHECKPOINT_DIR"
mkdir -p /scratch/ramanagarajayaram.k/logs

cd "$REPO_DIR"

# --- Self-chaining -------------------------------------------------------
# A single SLURM job only gets up to --time before it's killed, which may
# not be enough to reach --max_epochs. So instead of requesting one huge
# job, this script chains itself: it resubmits a copy of itself (dependent
# on this job finishing) BEFORE it starts training, then cancels that
# pending copy afterward if training turned out to be fully complete.
# train.py signals "fully complete" by writing a TRAINING_COMPLETE marker
# file next to the checkpoint once it reaches --max_epochs (see
# train.py's `completed_all_epochs` logic).
if [ -f "$CHECKPOINT_DIR/TRAINING_COMPLETE" ]; then
    echo "Training already complete"
    exit 0
fi

# Pre-submit the next link in the chain now, dependent on this job
# finishing (successfully or not) via afterany. If this job's training
# run finishes the whole schedule (max_epochs reached), we'll cancel this
# pending job below instead of letting it run needlessly.
NEXT_JOB_ID=$(sbatch --dependency=afterany:$SLURM_JOB_ID --parsable train.slurm)
echo "Submitted next job: $NEXT_JOB_ID"

# --- Train ---------------------------------------------------------------
# Single GPU, plain python (no torchrun) -- validated to work correctly in
# single-GPU mode; --resume picks up model/optimizer/epoch/global_step
# from the last checkpoint written by a previous link in the chain, if any.
python train.py \
    --dataset flickr \
    --max_epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --resume \
    --save_path "$CHECKPOINT_DIR/best.pt" \
    --save_every 500

# --- Post-training: stop the chain if we're actually done ----------------
if [ -f "$CHECKPOINT_DIR/TRAINING_COMPLETE" ]; then
    echo "Training complete, cancelling next job"
    scancel "$NEXT_JOB_ID"
else
    echo "Not complete, next job $NEXT_JOB_ID will continue."
fi
