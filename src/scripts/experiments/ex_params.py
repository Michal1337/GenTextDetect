from typing import Dict, List, Union

SEED: int = 1337

DATA_HUMAN_PATH: str = "../../../data/data_human/"
DATA_AI_PATH: str = "../../../data/data_ai/"

STATS_PATH: str = "../../../data/stats/"
MASTER_STATS_PATH: str = "../../../data/stats/data_stats_master.csv"

DATASETS_PATH: str = "../../../data/datasets/"
TRAINING_HISTORY_PATH: str = "../../../logs/"
CHECKPOINTS_PATH: str = "../../../checkpoints/"


PAD_TOKENS: Dict[str, str] = {
    "Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>",
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "<|finetune_right_pad_id|>",
    "Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>",
    "Mistral-Nemo-Instruct-2407": "<pad>",
    "Ministral-8B-Instruct-2410": "<pad>",
}

BASELINE_MODELS: Dict[str, Dict[str, int]] = {
    "mini": {
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "max_len": 512,
    },
    "small": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_len": 512,
    },
    "medium": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_len": 512,
    },
    "large": {
        "d_model": 1536,
        "num_layers": 36,
        "num_heads": 24,
        "max_len": 512,
    },
}

NUM_TOKENS_DETECT_LLM: int = 100_000_000
NUM_TOKENS_DETECT_LLM_FAMILY: int = 100_000_000

DATASETS: Dict[str, Dict[str, Union[int, bool, List[str]]]] = {
    "master_testset": {
        "num_tokens": 100_000,
        "cols_c0": ["human"],
        "reverse_labels": False,
    },
    "master_mini": {
        "num_tokens": 100_000,
        "cols_c0": ["human"],
        "reverse_labels": False,
    },
    "master_small": {
        "num_tokens": 1_000_000,
        "cols_c0": ["human"],
        "reverse_labels": False,
    },
    "master_medium": {
        "num_tokens": 10_000_000,
        "cols_c0": ["human"],
        "reverse_labels": False,
    },
    "master_large": {
        "num_tokens": 100_000_000,
        "cols_c0": ["human"],
        "reverse_labels": False,
    },
    # Meta
    "detect_Llama-3.1-8B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Llama-3.1-8B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Meta-Llama-3.1-70B-Instruct-AWQ-INT4": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Meta-Llama-3.1-70B-Instruct-AWQ-INT4"],
        "reverse_labels": True,
    },
    "detect_Llama-3.2-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Llama-3.2-3B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Meta-Llama-3.3-70B-Instruct-AWQ-INT4": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Meta-Llama-3.3-70B-Instruct-AWQ-INT4"],
        "reverse_labels": True,
    },
    # Microsoft
    "detect_Phi-3-mini-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-mini-128k-instruct"],
        "reverse_labels": True,
    },
    "detect_Phi-3-small-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-small-128k-instruct"],
        "reverse_labels": True,
    },
    "detect_Phi-3-medium-128k-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3-medium-128k-instruct"],
        "reverse_labels": True,
    },
    "detect_Phi-3.5-mini-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-3.5-mini-instruct"],
        "reverse_labels": True,
    },
    "detect_Phi-4-mini-instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Phi-4-mini-instruct"],
        "reverse_labels": True,
    },
    "detect_phi-4": {"num_tokens": NUM_TOKENS_DETECT_LLM, "cols_c0": ["phi-4"]},
    # Mistral
    "detect_Mistral-Nemo-Instruct-2407": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Mistral-Nemo-Instruct-2407"],
        "reverse_labels": True,
    },
    "detect_Ministral-8B-Instruct-2410": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Ministral-8B-Instruct-2410"],
        "reverse_labels": True,
    },
    # Qwen
    "detect_Qwen2-72B-Instruct-AWQ": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2-72B-Instruct-AWQ"],
        "reverse_labels": True,
    },
    "detect_Qwen2-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Qwen2.5-72B-Instruct-AWQ": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-72B-Instruct-AWQ"],
        "reverse_labels": True,
    },
    "detect_Qwen2.5-14B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-14B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Qwen2.5-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Qwen2.5-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Qwen2.5-3B-Instruct"],
        "reverse_labels": True,
    },
    # Falcon
    "detect_Falcon3-7B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Falcon3-7B-Instruct"],
        "reverse_labels": True,
    },
    "detect_Falcon3-3B-Instruct": {
        "num_tokens": NUM_TOKENS_DETECT_LLM,
        "cols_c0": ["Falcon3-3B-Instruct"],
        "reverse_labels": True,
    },
    # family detection
    "detect_llama_family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": [
            "Llama-3.1-8B-Instruct",
            "Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "Llama-3.2-3B-Instruct",
            "Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        ],
        "reverse_labels": True,
    },
    "detect_phi_family": {
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
    "detect_mistral_family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": ["Mistral-Nemo-Instruct-2407", "Ministral-8B-Instruct-2410"],
        "reverse_labels": True,
    },
    "detect_qwen_family": {
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
    "detect_falcon_family": {
        "num_tokens": NUM_TOKENS_DETECT_LLM_FAMILY,
        "cols_c0": ["Falcon3-7B-Instruct", "Falcon3-3B-Instruct"],
        "reverse_labels": True,
    },
}
