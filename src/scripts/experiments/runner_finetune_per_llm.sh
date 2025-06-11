#!/bin/bash
#SBATCH --job-name=per_llm      # Job name
#SBATCH --output=perllm2.log           # Standard output log
#SBATCH --error=perllm2.log             # Standard error log
#SBATCH --time=120:00:00               # Time limit
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=96G                     # Memory request
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=long              # Specify the long queue

# Ensure pyenv is initialized
export HOME=/mnt/evafs/faculty/home/mgromadzki
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Run Python script

# Meta
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.1-8B-Instruct detect-Llama-3.1-8B-Instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29535 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.2-3B-Instruct detect-Llama-3.2-3B-Instruct 5 8

# Microsoft
# torchrun --nproc_per_node=2 --master_port=29504 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-mini-128k-instruct detect-Phi-3-mini-128k-instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29510 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-small-128k-instruct detect-Phi-3-small-128k-instruct 5 1
# torchrun --nproc_per_node=2 --master_port=29511 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-medium-128k-instruct detect-Phi-3-medium-128k-instruct 5 1
# torchrun --nproc_per_node=2 --master_port=29535 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3.5-mini-instruct detect-Phi-3.5-mini-instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29536 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-4-mini-instruct detect-Phi-4-mini-instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29511 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/phi-4 detect-phi-4 5 1

# Mistral
# torchrun --nproc_per_node=2 --master_port=29510 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Mistral-Nemo-Instruct-2407 detect-Mistral-Nemo-Instruct-2407 5 2
# torchrun --nproc_per_node=2 --master_port=29510 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Ministral-8B-Instruct-2410 detect-Ministral-8B-Instruct-2410 5 4

# Qwen
# torchrun --nproc_per_node=2 --master_port=29501 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2-7B-Instruct detect-Qwen2-7B-Instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29511 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-14B-Instruct detect-Qwen2.5-14B-Instruct 5 2
# torchrun --nproc_per_node=2 --master_port=29510 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-7B-Instruct detect-Qwen2.5-7B-Instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29510 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-3B-Instruct detect-Qwen2.5-3B-Instruct 5 8

# Falcon
# torchrun --nproc_per_node=2 --master_port=29529 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-7B-Instruct detect-Falcon3-7B-Instruct 5 4
# torchrun --nproc_per_node=2 --master_port=29535 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-3B-Instruct detect-Falcon3-3B-Instruct 5 8

