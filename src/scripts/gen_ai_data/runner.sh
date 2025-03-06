#!/bin/bash
#SBATCH --job-name=vllm_inference      # Job name
#SBATCH --output=output.log           # Standard output log
#SBATCH --error=error.log             # Standard error log
#SBATCH --time=48:00:00               # Time limit (48 hours)
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=64G                     # Memory request (64GB)
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=long              # Specify the long queue

# Ensure pyenv is initialized
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run your Python script
python main.py
