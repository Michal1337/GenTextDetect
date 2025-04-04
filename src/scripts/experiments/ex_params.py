from typing import Dict

STATS_PATH: str = "../../../data/stats/"
MASTER_STATS_PATH: str = "../../../data/stats/data_stats_master.csv"

DATASETS_PATH = "../../../data/datasets/"

DATASETS = {
    "master_mini": (100_000, ["human"]),
    "master_small": (1_000_000, ["human"]),
    "master_medium": (10_000_000, ["human"]),
    "master_large": (100_000_000, ["human"]),
}

BASELINE_MODELS = {
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
