import csv
import random
from itertools import islice
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gen_params import *

random.seed(SEED)


def batchify(iterable: Iterable[str], batch_size: int):
    """Splits an iterable into smaller batches."""
    iterable = iter(iterable)
    while batch := list(islice(iterable, batch_size)):
        yield batch


def save_to_csv(
    path: str,
    prompts: List[str],
    responses: List[str],
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    """Saves prompts, responses and sampling parameters to a CSV file."""
    with open(path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response, temperature, top_p, top_k])


def generate_responses(
    model: LLM, prompts: List[str], sampling_params: SamplingParams
) -> List[str]:
    """Generate a batch of outputs using vLLM with customizable sampling parameters."""
    outputs = model.chat(prompts, sampling_params=sampling_params, use_tqdm=False)

    return [remove_surrounding_quotes(sample.outputs[0].text) for sample in outputs]


def remove_surrounding_quotes(s: str) -> str:
    """
    Removes single or double quotes from the start and end of the string if they exist.
    """
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def check_for_too_long_prompts(
    df: pd.DataFrame, prompts: List[Dict[str, str]], max_tokens_prompt: int
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Check if any of the prompts are too long."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    lens = []
    batch_size = 128

    print("Tokenizing prompts...")
    for prompts_batch in tqdm(
        batchify(prompts, batch_size), total=len(prompts) // batch_size
    ):
        tokens = tokenizer.apply_chat_template(prompts_batch)
        lens.extend([len(token) for token in tokens])

    too_long = [idx for idx, length in enumerate(lens) if length > max_tokens_prompt]
    df.drop(too_long, inplace=True)
    prompts = [prompts[i] for i in range(len(prompts)) if i not in too_long]
    print("Removed too long prompts:", len(too_long))

    return df, prompts


def generate_texts(prompts, llms, sampling_params, batch_size, base_path):
    for llm, quant in llms:
        model = LLM(model=llm, dtype="half", max_model_len=10_000, quantization=quant)
        csv_path = f"{base_path}{llm.split('/')[-1]}.csv"

        # init csv file
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["prompt", "response", "temperature", "top_p", "top_k"])

        cnt = 0
        print(f"Generating texts for {llm}...")
        for prompts_batch in tqdm(
            batchify(prompts, batch_size), total=len(prompts) // batch_size
        ):
            params = random.choice(sampling_params)
            responses = generate_responses(model, prompts_batch, params)
            save_to_csv(
                csv_path,
                prompts_batch,
                responses,
                params.temperature,
                params.top_p,
                params.top_k,
            )
            cnt += 1
            if cnt > 2:
                break

        df = pd.read_csv(csv_path)
        print(
            f"Expected samples: {len(prompts)}, Actual samples: {len(df)}, Match: {len(prompts) == len(df)}, Model: {llm}"
        )
