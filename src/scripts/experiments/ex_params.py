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