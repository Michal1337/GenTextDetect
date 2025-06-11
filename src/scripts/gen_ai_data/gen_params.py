from typing import List, Optional, Tuple

from vllm import SamplingParams

RAW_DATA_BASE_PATH: str = "../../../data/data_raw/"
HUMAN_DATA_BASE_PATH: str = "../../../data/data_human/"
AI_DATA_BASE_PATH: str = "../../../data/data_ai/"

SEED: int = 1337
MAX_TOKENS_PROMPT: int = 16_384
MAX_TOKENS_GENERATE: int = 16_384
MAX_MODEL_LEN: int = MAX_TOKENS_PROMPT + MAX_TOKENS_GENERATE

MAX_WORKERS: int = 256
MAX_RETRIES: int = 5
API_KEY: str = "your api key here"


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


MODEL_PATH: str = "/mnt/evafs/groups/re-com/mgromadzki/llms/"

LLMS: List[Tuple[str, str, Optional[str]]] = [
    # Meta
    (
        "meta-llama/Llama-3.1-8B-Instruct",
        MODEL_PATH + "meta-llama/Llama-3.1-8B-Instruct",
        None,
    ),
    (
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        MODEL_PATH + "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "awq",
    ),
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        MODEL_PATH + "meta-llama/Llama-3.2-3B-Instruct",
        None,
    ),
    (
        "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        MODEL_PATH + "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "awq",
    ),
    # Microsoft
    (
        "microsoft/Phi-3-mini-128k-instruct",
        MODEL_PATH + "microsoft/Phi-3-mini-128k-instruct",
        None,
    ),
    (
        "microsoft/Phi-3-small-128k-instruct",
        MODEL_PATH + "microsoft/Phi-3-small-128k-instruct",
        None,
    ),
    (
        "microsoft/Phi-3-medium-128k-instruct",
        MODEL_PATH + "microsoft/Phi-3-medium-128k-instruct",
        None,
    ),
    (
        "microsoft/Phi-3.5-mini-instruct",
        MODEL_PATH + "microsoft/Phi-3.5-mini-instruct",
        None,
    ),
    (
        "microsoft/Phi-4-mini-instruct",
        MODEL_PATH + "microsoft/Phi-4-mini-instruct",
        None,
    ),
    ("microsoft/phi-4", MODEL_PATH + "microsoft/phi-4", None),
    # Mistral
    (
        "mistralai/Mistral-Nemo-Instruct-2407",
        MODEL_PATH + "mistralai/Mistral-Nemo-Instruct-2407",
        None,
    ),
    (
        "mistralai/Ministral-8B-Instruct-2410",
        MODEL_PATH + "mistralai/Ministral-8B-Instruct-2410",
        None,
    ),
    # # Qwen
    ("Qwen/Qwen2-72B-Instruct-AWQ", MODEL_PATH + "Qwen/Qwen2-72B-Instruct-AWQ", "awq"),
    ("Qwen/Qwen2-7B-Instruct", MODEL_PATH + "Qwen/Qwen2-7B-Instruct", None),
    (
        "Qwen/Qwen2.5-72B-Instruct-AWQ",
        MODEL_PATH + "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "awq",
    ),
    ("Qwen/Qwen2.5-14B-Instruct", MODEL_PATH + "Qwen/Qwen2.5-14B-Instruct", None),
    ("Qwen/Qwen2.5-7B-Instruct", MODEL_PATH + "Qwen/Qwen2.5-7B-Instruct", None),
    ("Qwen/Qwen2.5-3B-Instruct", MODEL_PATH + "Qwen/Qwen2.5-3B-Instruct", None),
    # Falcon
    ("tiiuae/Falcon3-7B-Instruct", MODEL_PATH + "tiiuae/Falcon3-7B-Instruct", None),
    ("tiiuae/Falcon3-3B-Instruct", MODEL_PATH + "tiiuae/Falcon3-3B-Instruct", None),
    # OpenAI
    ("gpt-4.1-nano-2025-04-14", "proprietary", None),
]
