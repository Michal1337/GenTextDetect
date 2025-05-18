#!/bin/bash
#SBATCH --job-name=ft2      # Job name
#SBATCH --output=output_ft2.log           # Standard output log
#SBATCH --error=error_ft2.log             # Standard error log
#SBATCH --time=120:00:00               # Time limit
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=96G                     # Memory request
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=long              # Specify the long queue

# Ensure pyenv is initialized
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script

# Meta
# torchrun --nproc_per_node=2 --master_port=29505 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.1-8B-Instruct master-large 5 4
# torchrun --nproc_per_node=2 --master_port=29505 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.2-3B-Instruct master-large 5 8

# # Microsoft
# torchrun --nproc_per_node=2 --master_port=29504 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-mini-128k-instruct master-large 5 2
# torchrun --nproc_per_node=2 --master_port=29504 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-small-128k-instruct master-large 5 1
# torchrun --nproc_per_node=2 --master_port=29504 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-medium-128k-instruct master-large 5 1
# torchrun --nproc_per_node=2 --master_port=29503 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3.5-mini-instruct master-large 5 2
# torchrun --nproc_per_node=2 --master_port=29503 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-4-mini-instruct master-large 5 2
# torchrun --nproc_per_node=2 --master_port=29503 training_finetune.py /mnt/evafs/groups/re-c'om/mgromadzki/llms/microsoft/phi-4 master-large 5 1

# # Mistral
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Mistral-Nemo-Instruct-2407 master-large 5 2
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Ministral-8B-Instruct-2410 master-large 5 4
# # Qwen
torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2-7B-Instruct master-large 5 4
torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-14B-Instruct master-large 5 2
torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-7B-Instruct master-large 5 4
torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-3B-Instruct master-large 5 8

# Falcon
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-7B-Instruct master-large 5 4
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-3B-Instruct master-large 5 8


