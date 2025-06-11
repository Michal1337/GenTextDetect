#!/bin/bash
#SBATCH --job-name=eval     # Job name
#SBATCH --output=eval1.log           # Standard output log
#SBATCH --error=eval1.log             # Standard error log
#SBATCH --time=24:00:00               # Time limit
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G                     # Memory request
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=short              # Specify the long queue
#SBATCH --nodelist=dgx-4

# Ensure pyenv is initialized
export HOME=/mnt/evafs/faculty/home/mgromadzki
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
# Run Python script

# python evaluation_noddp.py
torchrun --nproc_per_node=2 evaluation.py


