#!/bin/bash

#SBATCH --account=hai_1179
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=../logs/%j.out
#SBATCH --job-name=GUROMPL


curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

module load uv
module load Stages/2025
module load CUDA/12
source ../multimodal_memorization/.venv/bin/activate
echo "Using Python from: $(which python)"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export HF_HUB_OFFLINE=1
export WANDB_DISABLED=1
jutil env activate -p hai_1179
SCRIPT=$PROJECT/p4ms_hackathon_warsaw_code/bash_scripts/train/vqa.bash

curl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
mkdir -p ffmpeg-release &&
tar Jxf ffmpeg-release-amd64-static.tar.xz --strip-components=1 -C ffmpeg-release &&
PATH=$(readlink -f ffmpeg-release):$PATH

bash "$SCRIPT" "$@"
echo "Job finished"