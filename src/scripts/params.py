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
