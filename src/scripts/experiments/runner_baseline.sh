#!/bin/bash
#SBATCH --job-name=tmp      # Job name
#SBATCH --output=tmp.log           # Standard output log
#SBATCH --error=tmp.log             # Standard error log
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

# torchrun --nproc_per_node=2 training_baseline.py mini master-mini 10 32
# torchrun --nproc_per_node=2 training_baseline.py mini master-small 10 32
# torchrun --nproc_per_node=2 training_baseline.py mini master-medium 10 32
# torchrun --nproc_per_node=2 training_baseline.py mini master-large 10 32

# torchrun --nproc_per_node=2 --master_port=29501 training_baseline.py small master-mini 10 16
# torchrun --nproc_per_node=2 --master_port=29501 training_baseline.py small master-small 10 16
# torchrun --nproc_per_node=2 --master_port=29501 training_baseline.py small master-medium 10 16
# torchrun --nproc_per_node=2 --master_port=29501 training_baseline.py small master-large 10 16

# torchrun --nproc_per_node=2 training_baseline.py medium master-mini 10 8
# torchrun --nproc_per_node=2 training_baseline.py medium master-small 10 8
# torchrun --nproc_per_node=2 training_baseline.py medium master-medium 10 8
# torchrun --nproc_per_node=2 training_baseline.py medium master-large 10 8

# torchrun --nproc_per_node=2 --master_port=29502 training_baseline.py large master-mini 10 4
# torchrun --nproc_per_node=2 --master_port=29502 training_baseline.py large master-small 10 4
# torchrun --nproc_per_node=2 --master_port=29502 training_baseline.py large master-medium 10 4
# torchrun --nproc_per_node=2 --master_port=29505 training_baseline.py large master-large 10 4

torchrun --nproc_per_node=2 --master_port=29525 training_baseline.py mini detect-gpt-4.1-nano-2025-04-14 10 32
