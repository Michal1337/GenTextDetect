from typing import List

HF_DS_NAMES: List[str] = [
    "liamdugan/raid",
    "EdinburghNLP/xsum",
    "euclaise/writingprompts",
    "google-research-datasets/natural_questions",
]

DATA_HUMAN_PATH: str = "../../data/data_human/"
DATA_AI_PATH: str = "../../data/data_ai/"

STATS_PATH: str = "../../data/stats/"
MASTER_STATS_PATH: str = "../../data/stats/data_stats_master.csv"

FEATURES_PATH: str = "../../data/features/"
FEATURES_STATS_PATH: str = "../../data/features/features_stats_master.csv"

NGRAMS_PATH: str = "../../data/ngrams/"
MIN_NGRAM_LEVEL: int = 1
MAX_NGRAM_LEVEL: int = 4

DATASET_IDX_PATH = "../../data/datasets/idx/"

MAIN_DATASET_SIZES = {
    "mini": 100_000,
    "small": 1_000_000,
    "medium": 10_000_000,
    "large": 100_000_000,
}
