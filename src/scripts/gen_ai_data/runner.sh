#!/bin/bash
#SBATCH --job-name=vllm_inference      # Job name
#SBATCH --output=output6.log           # Standard output log
#SBATCH --error=error6.log             # Standard error log
#SBATCH --time=24:00:00               # Time limit
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=96G                     # Memory request
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=short              # Specify the long queue

# Ensure pyenv is initialized
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script
python main.py
