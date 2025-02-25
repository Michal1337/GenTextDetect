from typing import List, Tuple

from vllm import SamplingParams

RAW_DATA_BASE_PATH: str = "../../../data/data_raw/"
HUMAN_DATA_BASE_PATH: str = "../../../data/data_human/"
AI_DATA_BASE_PATH: str = "../../../data/data_ai/"

SEED: int = 1337
MAX_TOKENS_PROMPT: int = 32_768
MAX_TOKENS_GENERATE: int = 30_000


SAMPLING_PARAMS: List[SamplingParams] = [
    SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Pure Greedy (fully deterministic)
    SamplingParams(
        temperature=0.2, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Highly Deterministic
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
        temperature=0.9, top_p=0.8, top_k=40, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Slightly More Diverse but Coherent
    SamplingParams(
        temperature=1.0, top_p=0.95, top_k=30, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Default Creative Mode
    SamplingParams(
        temperature=1.2, top_p=0.7, top_k=20, max_tokens=MAX_TOKENS_GENERATE, seed=SEED
    ),  # Highly Creative
]

LLMS: List[Tuple[str, str]] = [("meta-llama/Llama-3.2-1B-Instruct", None)]
