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


MODEL_PATH = "/path/to/models/"

LLMS: List[Tuple[str, Optional[str], str]] = [
    # Meta
    ("meta-llama/Llama-3.1-8B-Instruct", None, MODEL_PATH + "meta-llama/Llama-3.1-8B-Instruct"),
    ("unsloth/Meta-Llama-3.1-70B-bnb-4bit", "bitsandbytes", MODEL_PATH + "unsloth/Meta-Llama-3.1-70B-bnb-4bit"),
    ("meta-llama/Llama-3.2-3B-Instruct", None, MODEL_PATH + "meta-llama/Llama-3.2-3B-Instruct"),
    ("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "bitsandbytes", MODEL_PATH + "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
    
    # Microsoft
    ("microsoft/Phi-3-mini-128k-instruct", None, MODEL_PATH + "microsoft/Phi-3-mini-128k-instruct"),
    ("microsoft/Phi-3-small-128k-instruct", None, MODEL_PATH + "microsoft/Phi-3-small-128k-instruct"),
    ("microsoft/Phi-3-medium-128k-instruct", None, MODEL_PATH + "microsoft/Phi-3-medium-128k-instruct"),
    ("microsoft/Phi-3.5-mini-instruct", None, MODEL_PATH + "microsoft/Phi-3.5-mini-instruct"),
    ("microsoft/Phi-4-mini-instruct", None, MODEL_PATH + "microsoft/Phi-4-mini-instruct"),
    ("microsoft/phi-4", None, MODEL_PATH + "microsoft/phi-4"),
    
    # Mistral
    ("mistralai/Mistral-Nemo-Instruct-2407", None, MODEL_PATH + "mistralai/Mistral-Nemo-Instruct-2407"),
    ("mistralai/Ministral-8B-Instruct-2410", None, MODEL_PATH + "mistralai/Ministral-8B-Instruct-2410"),
    
    # Qwen
    ("Qwen/Qwen2-72B-Instruct-AWQ", "awq", MODEL_PATH + "Qwen/Qwen2-72B-Instruct-AWQ"),
    ("Qwen/Qwen2-7B-Instruct", None, MODEL_PATH + "Qwen/Qwen2-7B-Instruct"),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", "awq", MODEL_PATH + "Qwen/Qwen2.5-72B-Instruct-AWQ"),
    ("Qwen/Qwen2.5-14B-Instruct", None, MODEL_PATH + "Qwen/Qwen2.5-14B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", None, MODEL_PATH + "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", None, MODEL_PATH + "Qwen/Qwen2.5-3B-Instruct"),
    
    # Falcon
    ("tiiuae/Falcon3-7B-Instruct", None, MODEL_PATH + "tiiuae/Falcon3-7B-Instruct"),
    ("tiiuae/Falcon3-3B-Instruct", None, MODEL_PATH + "tiiuae/Falcon3-3B-Instruct"),
]

