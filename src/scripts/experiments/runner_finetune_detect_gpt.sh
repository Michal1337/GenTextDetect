#!/bin/bash
#SBATCH --job-name=detect_gpt      # Job name
#SBATCH --output=gpt.log           # Standard output log
#SBATCH --error=gpt.log             # Standard error log
#SBATCH --time=120:00:00               # Time limit
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --mem=96G                     # Memory request
#SBATCH --cpus-per-task=8             # Allocate CPU cores
#SBATCH --partition=long              # Specify the long queue
#SBATCH --nodelist=dgx-4


export HOME=/mnt/evafs/faculty/home/mgromadzki
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Meta
# torchrun --nproc_per_node=2 --master_port=29521 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.1-8B-Instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29521 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/meta-llama/Llama-3.2-3B-Instruct detect-gpt-4.1-nano-2025-04-14 5 8

# Microsoft
# torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-mini-128k-instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29543 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-small-128k-instruct detect-gpt-4.1-nano-2025-04-14 5 1
# torchrun --nproc_per_node=2 --master_port=29544 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3-medium-128k-instruct detect-gpt-4.1-nano-2025-04-14 5 1
# torchrun --nproc_per_node=2 --master_port=29502 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-3.5-mini-instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29530 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/Phi-4-mini-instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29516 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/microsoft/phi-4 detect-gpt-4.1-nano-2025-04-14 5 1

# Mistral
# torchrun --nproc_per_node=2 --master_port=29540 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Mistral-Nemo-Instruct-2407 detect-gpt-4.1-nano-2025-04-14 5 2
# torchrun --nproc_per_node=2 --master_port=29503 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/mistralai/Ministral-8B-Instruct-2410 detect-gpt-4.1-nano-2025-04-14 5 4

# Qwen
# torchrun --nproc_per_node=2 --master_port=29523 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2-7B-Instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29515 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-14B-Instruct detect-gpt-4.1-nano-2025-04-14 5 2
# torchrun --nproc_per_node=2 --master_port=29524 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-7B-Instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29524 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/Qwen/Qwen2.5-3B-Instruct detect-gpt-4.1-nano-2025-04-14 5 8

# Falcon
# torchrun --nproc_per_node=2 --master_port=29522 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-7B-Instruct detect-gpt-4.1-nano-2025-04-14 5 4
# torchrun --nproc_per_node=2 --master_port=29522 training_finetune.py /mnt/evafs/groups/re-com/mgromadzki/llms/tiiuae/Falcon3-3B-Instruct detect-gpt-4.1-nano-2025-04-14 5 8



