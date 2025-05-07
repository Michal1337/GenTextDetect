#!/bin/bash
#SBATCH --job-name=ngrams       # Job name
#SBATCH --output=output.log           # Standard output log
#SBATCH --error=error.log             # Standard error log
#SBATCH --time=24:00:00                # Time limit
#SBATCH --mem=64G                      # Memory request
#SBATCH --cpus-per-task=2             # Allocate CPU cores
#SBATCH --partition=short              # Specify the short queue

# Ensure pyenv is initialized
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script
python ngrams_bos.py
