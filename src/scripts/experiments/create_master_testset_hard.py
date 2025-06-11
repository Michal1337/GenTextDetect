import argparse
import csv
import os
import random
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ex_params import (
    DATASETS_PATH,
    MAX_MODEL_LEN,
    MAX_TOKENS_GENERATE,
    NUM_SAMPLES,
    SEED,
)

random.seed(SEED)

BATCH_SIZE = 32
MODEL_PATH: str = "/mnt/evafs/groups/re-com/mgromadzki/llms/"
LLMS: List[Tuple[str, str, Optional[str]]] = [
    (
        "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        MODEL_PATH + "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "awq",
    ),
    (
        "mistralai/Mistral-Nemo-Instruct-2407",
        MODEL_PATH + "mistralai/Mistral-Nemo-Instruct-2407",
        None,
    ),
    ("Qwen/Qwen2.5-14B-Instruct", MODEL_PATH + "Qwen/Qwen2.5-14B-Instruct", None),
    ("tiiuae/Falcon3-7B-Instruct", MODEL_PATH + "tiiuae/Falcon3-7B-Instruct", None),
    ("microsoft/phi-4", MODEL_PATH + "microsoft/phi-4", None),
]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for rewritting texts to make them more human. Based on the provided text, rewrite it to make it sound more human, keep the meaning the same. MAKE SURE TO REPLY ONLY WITH THE REWRITTEN TEXT.",
    },
    {"role": "user", "content": "Text:\n{text}"},
    {"role": "assistant", "content": "Rewritten text:\n"},
]


def batchify(iterable: Iterable[str], batch_size: int):
    """Splits an iterable into smaller batches."""
    iterable = iter(iterable)
    while batch := list(islice(iterable, batch_size)):
        yield batch


def save_to_csv(
    path: str,
    prompts: List[str],
    responses: List[str],
) -> None:
    """Saves prompts, responses and sampling parameters to a CSV file."""
    with open(path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response])


def generate_responses(
    model: LLM, prompts: List[str], sampling_params: SamplingParams
) -> List[str]:
    """Generate a batch of outputs using vLLM with customizable sampling parameters."""
    outputs = model.chat(
        prompts,
        sampling_params=sampling_params,
        add_generation_prompt=False,
        continue_final_message=True,
        use_tqdm=False,
    )

    return [preproc_response(sample.outputs[0].text) for sample in outputs]


def preproc_response(s: str) -> str:
    """
    Removes single or double quotes from the start and end of the string if they exist and removes leading newlines and spaces.
    """
    s = s.lstrip("\n ")

    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def generate_texts(
    prompts: List[List[Dict[str, str]]],
    df: pd.DataFrame,
    llm_name: str,
    llm_path: str,
    quant: str,
    batch_size: int,
    csv_path: str,
) -> None:
    model = LLM(
        model=llm_path,
        quantization=quant,
        max_model_len=(
            MAX_MODEL_LEN // 2 if llm_name == "microsoft/phi-4" else MAX_MODEL_LEN
        ),
        trust_remote_code=True,
        seed=SEED,
        tensor_parallel_size=2,
    )

    # init csv file
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "text"])

    batches = list(batchify(prompts, batch_size))
    print(f"Generating texts for {llm_name}...")
    for prompts_batch in tqdm(batches, total=len(prompts) // batch_size):
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=MAX_TOKENS_GENERATE,
            seed=SEED,
        )
        responses = generate_responses(model, prompts_batch, params)

        save_to_csv(
            csv_path,
            prompts_batch,
            responses,
        )

    df_new = pd.read_csv(csv_path)
    df_new["label"] = df["label"]
    df_new["data"] = df["data"]
    df_new["model"] = df["model"]
    df_new.dropna(inplace=True)
    df_new.to_csv(csv_path, index=False)
    print(
        f"Expected samples: {len(prompts)}, Actual samples: {len(df_new)}, Match: {len(prompts) == len(df_new)}, Model: {llm_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create master testset-hard")
    parser.add_argument("iter", type=int, help="Iteration number")
    args = parser.parse_args()

    if args.iter == 0:
        master_testset_path = DATASETS_PATH + "master-testset/test.csv"
        os.mkdir(DATASETS_PATH + "master-testset-hard")
        csv_path = DATASETS_PATH + "master-testset-hard/test0.csv"

        df = pd.read_csv(master_testset_path)
        df = df[df["label"] == 1]
        df = df.sample(n=NUM_SAMPLES, random_state=SEED).reset_index(drop=True)

        df.to_csv(csv_path, index=False)

    else:
        csv_path_old = DATASETS_PATH + f"master-testset-hard/test{args.iter-1}.csv"
        csv_path = DATASETS_PATH + f"master-testset-hard/test{args.iter}.csv"

        df = pd.read_csv(csv_path_old)
        df.dropna(inplace=True)

        prompts = [
            [
                BASE_PROMPT[0],
                {
                    "role": "user",
                    "content": BASE_PROMPT[1]["content"].format(text=text),
                },
                BASE_PROMPT[2],
            ]
            for text in df["text"].values
        ]

        llm_name, llm_path, quant = LLMS[args.iter - 1]

        generate_texts(
            prompts,
            df,
            llm_name,
            llm_path,
            quant,
            BATCH_SIZE,
            csv_path,
        )
