from typing import Dict, List, Union

SEED: int = 1337

DATA_HUMAN_PATH: str = "../../../data/data_human/"
DATA_AI_PATH: str = "../../../data/data_ai/"

DATA_HUMAN_FEATURES_PATH: str = "../../../data/features/data_human/"
DATA_AI_FEATURES_PATH: str = "../../../data/features/data_ai/"

STATS_PATH: str = "../../../data/stats/"

DATASETS_PATH: str = "../../../data/datasets/"
TRAINING_HISTORY_PATH: str = "../../../logs/"
CHECKPOINTS_PATH: str = "../../../checkpoints/"
PREDICTIONS_PATH: str = "../../../predictions/"

# Params for master-testset-hard
NUM_SAMPLES: int = 10_000
MAX_MODEL_LEN: int = 32_768
MAX_TOKENS_GENERATE: int = 16_384

# Maximum text length for LLMs finetuning
MAX_TEXT_LENGTH: int = 8192

MODEL_PATH: str = "/mnt/evafs/groups/re-com/mgromadzki/llms/"

SELECTED_FEATURES1: List[str] = [
    "noun_ratio",
    "automated_readability_index",
    "contraction_count",
    "discourse_marker_ratio",
]

SELECTED_FEATURES2: List[str] = [
    "entropy_score",
    "conjunction_count",
    "syntactic_depth",
    "Mass",
]

PAD_TOKENS: Dict[str, str] = {
    "Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>",
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "<|finetune_right_pad_id|>",
    "Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>",
    "Mistral-Nemo-Instruct-2407": "<pad>",
    "Ministral-8B-Instruct-2410": "<pad>",
}

BASELINE_MODELS: Dict[str, Dict[str, Union[int, float]]] = {
    "mini": {
        "d_model": 324,
        "num_layers": 4,
        "num_heads": 4,
        "max_len": 8192,
        "start_lr": 3e-4,
    },
    "small": {
        "d_model": 546,
        "num_layers": 8,
        "num_heads": 6,
        "max_len": 8192,
        "start_lr": 1e-4,
    },
    "medium": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_len": 8192,
        "start_lr": 5e-5,
    },
    "large": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_len": 8192,
        "start_lr": 2e-5,
    },
}

NUM_TOKENS_DETECT_LLM: int = 60_000_000
NUM_TOKENS_DETECT_LLM_FAMILY: int = 100_000_000

DATASETS: Dict[str, Dict[str, Union[int, bool, List[str]]]] = {
    # "master-testset": {
    #     "num_tokens": 100_000_000,
    #     "cols_c0": ["human"],
    #     "reverse_labels": False,
    # },
    # "master-mini": {
    #     "num_tokens": 10_000_000,
    #     "cols_c0": ["human"],
    #     "reverse_labels": False,
    # },
    # "master-small": {
    #     "num_tokens": 20_000_000,
    #     "cols_c0": ["human"],
    #     "reverse_labels": False,
    # },
    # "master-medium": {
    #     "num_tokens": 50_000_000,
    #     "cols_c0": ["human"],
    #     "reverse_labels": False,
    # },
    # "master-large": {
    #     "num_tokens": 100_000_000,
    #     "cols_c0": ["human"],
    #     "reverse_labels": False,
    # },
    "detect-gpt-4.1-nano-2025-04-14-v2": {
        "num_tokens": 50_000_000,
        "cols_c0": ["gpt-4.1-nano-2025-04-14"],
        "reverse_labels": True,
    },
    # Meta
    "detect-Llama-3.1-8B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Llama-3.1-8B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Meta-Llama-3.1-70B-Instruct-AWQ-INT4": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Meta-Llama-3.1-70B-Instruct-AWQ-INT4"],
        "reverse_labels": True,
    },
    "detect-Llama-3.2-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Llama-3.2-3B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Meta-Llama-3.3-70B-Instruct-AWQ-INT4": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Meta-Llama-3.3-70B-Instruct-AWQ-INT4"],
        "reverse_labels": True,
    },
    # Microsoft
    "detect-Phi-3-mini-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-mini-128k-instruct"],
        "reverse_labels": True,
    },
    "detect-Phi-3-small-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-small-128k-instruct"],
        "reverse_labels": True,
    },
    "detect-Phi-3-medium-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-medium-128k-instruct"],
        "reverse_labels": True,
    },
    "detect-Phi-3.5-mini-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3.5-mini-instruct"],
        "reverse_labels": True,
    },
    "detect-Phi-4-mini-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-4-mini-instruct"],
        "reverse_labels": True,
    },
    "detect-phi-4": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["phi-4"],
        "reverse_labels": True,
    },
    # Mistral
    "detect-Mistral-Nemo-Instruct-2407": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Mistral-Nemo-Instruct-2407"],
        "reverse_labels": True,
    },
    "detect-Ministral-8B-Instruct-2410": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Ministral-8B-Instruct-2410"],
        "reverse_labels": True,
    },
    # Qwen
    "detect-Qwen2-72B-Instruct-AWQ": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2-72B-Instruct-AWQ"],
        "reverse_labels": True,
    },
    "detect-Qwen2-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Qwen2.5-72B-Instruct-AWQ": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-72B-Instruct-AWQ"],
        "reverse_labels": True,
    },
    "detect-Qwen2.5-14B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-14B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Qwen2.5-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Qwen2.5-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-3B-Instruct"],
        "reverse_labels": True,
    },
    # Falcon
    "detect-Falcon3-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Falcon3-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect-Falcon3-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Falcon3-3B-Instruct"],
        "reverse_labels": True,
    },
    # family detection
    "detect-llama-family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": [
            "Llama-3.1-8B-Instruct",
            "Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "Llama-3.2-3B-Instruct",
            "Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        ],
        "reverse_labels": True,
    },
    "detect-phi-family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": [
            "Phi-3-mini-128k-instruct",
            "Phi-3-small-128k-instruct",
            "Phi-3-medium-128k-instruct",
            "Phi-3.5-mini-instruct",
            "Phi-4-mini-instruct",
            "phi-4",
        ],
        "reverse_labels": True,
    },
    "detect-mistral-family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": ["Mistral-Nemo-Instruct-2407", "Ministral-8B-Instruct-2410"],
        "reverse_labels": True,
    },
    "detect-qwen-family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": [
            "Qwen2-72B-Instruct-AWQ",
            "Qwen2-7B-Instruct",
            "Qwen2.5-72B-Instruct-AWQ",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-3B-Instruct",
        ],
        "reverse_labels": True,
    },
    "detect-falcon-family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": ["Falcon3-7B-Instruct", "Falcon3-3B-Instruct"],
        "reverse_labels": True,
    },
}

TRAINING_CONFIG: Dict[str, Dict[str, Union[int, float]]] = {
    "Llama-3.1-8B-Instruct": {"start_lr": 2e-4, "total_batch_size": 64},
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": {"start_lr": 1e-4, "total_batch_size": 16},
    "Llama-3.2-3B-Instruct": {"start_lr": 5e-4, "total_batch_size": 128},
    "Meta-Llama-3.3-70B-Instruct-AWQ-INT4": {"start_lr": 1e-4, "total_batch_size": 16},
    "Phi-3-mini-128k-instruct": {"start_lr": 3e-4, "total_batch_size": 128},
    "Phi-3-small-128k-instruct": {"start_lr": 2e-4, "total_batch_size": 64},
    "Phi-3-medium-128k-instruct": {"start_lr": 1e-4, "total_batch_size": 32},
    "Phi-3.5-mini-instruct": {"start_lr": 3e-4, "total_batch_size": 128},
    "Phi-4-mini-instruct": {"start_lr": 3e-4, "total_batch_size": 128},
    "phi-4": {"start_lr": 1e-4, "total_batch_size": 32},
    "Mistral-Nemo-Instruct-2407": {"start_lr": 2e-4, "total_batch_size": 64},
    "Ministral-8B-Instruct-2410": {"start_lr": 2e-4, "total_batch_size": 64},
    "Qwen2-72B-Instruct-AWQ": {"start_lr": 1e-4, "total_batch_size": 16},
    "Qwen2-7B-Instruct": {"start_lr": 2e-4, "total_batch_size": 64},
    "Qwen2.5-72B-Instruct-AWQ": {"start_lr": 1e-4, "total_batch_size": 16},
    "Qwen2.5-14B-Instruct": {"start_lr": 1e-4, "total_batch_size": 32},
    "Qwen2.5-7B-Instruct": {"start_lr": 2e-4, "total_batch_size": 64},
    "Qwen2.5-3B-Instruct": {"start_lr": 5e-4, "total_batch_size": 128},
    "Falcon3-7B-Instruct": {"start_lr": 2e-4, "total_batch_size": 64},
    "Falcon3-3B-Instruct": {"start_lr": 5e-4, "total_batch_size": 128},
}
