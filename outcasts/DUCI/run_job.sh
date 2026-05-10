#!/bin/bash
#SBATCH --account=training2615
#SBATCH --job-name=duci_inference
#SBATCH --output=output/job_%j.log
#SBATCH --error=output/job_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


mkdir -p output

echo "Starting job on: $(hostname)"
echo "Project account: training2615"

nvidia-smi

echo "Activating virtual environment..."
source .venv/bin/activate

which python

echo "Starting Python script..."
python -u solve_duci_v2.py
echo "Script finished."