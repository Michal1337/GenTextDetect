from typing import List, Tuple, Optional

from vllm import SamplingParams

RAW_DATA_BASE_PATH: str = "../../../data/data_raw/"
HUMAN_DATA_BASE_PATH: str = "../../../data/data_human/"
AI_DATA_BASE_PATH: str = "../../../data/data_ai/"

SEED: int = 1337
MAX_TOKENS_PROMPT: int = 32_768
MAX_TOKENS_GENERATE: int = 50_000


SAMPLING_PARAMS: List[SamplingParams] = [
    SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Pure Greedy (fully deterministic)
    SamplingParams(
        temperature=0.5,
        top_p=0.95,
        top_k=100,
        max_tokens=MAX_TOKENS_GENERATE,
        seed=SEED,
    ),  # Mildly Deterministic but Flexible
    SamplingParams(
        temperature=0.7, top_p=0.9, top_k=50, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Balanced and Natural
    SamplingParams(
        temperature=1.0, top_p=0.95, top_k=30, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Default Creative Mode
]


LLMS: List[Tuple[str, Optional[str]]] = [("meta-llama/Llama-3.2-1B-Instruct", None)]
