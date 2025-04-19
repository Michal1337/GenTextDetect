import argparse
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, SAMPLING_PARAMS, SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

DS_NAME = "liamdugan/raid"
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "raid_human.csv"
AI_DATA_PATH = AI_DATA_BASE_PATH + "raid/raid_"

PROMPT_COLS = ["domain", "title"]
TEXT_COL = "generation"
TO_DROP = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation_length",
    "title_length",
]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant specializing in writing texts across various domains, including abstracts and news articles, based on provided titles. Based on the given domain and title, generate a text of appropriate length and style that aligns with the specified domain. MAKE SURE TO REPLY ONLY WITH THE GENERATED TEXT.",
    },
    {"role": "user", "content": "Domain:\n{domain}\nTitle:\n{title}"},
    {"role": "assistant", "content": "Generated text:\n"},
]

PERCENT_SAMPLE = 0.25
BATCH_SIZE = 64


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the dataset."""
    dataset = load_dataset(DS_NAME, "raid")["train"]
    dataset = dataset.filter(lambda x: x["model"] == "human")
    df = dataset.to_pandas()

    # Compute title and generation lengths
    df["title_length"] = df["title"].str.len()
    df["generation_length"] = df[TEXT_COL].str.len()

    # Filter out invalid rows
    df = df[(df["title_length"] >= 10) & (df["generation_length"] >= 50)]
    df.drop_duplicates(subset=[TEXT_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(domain=domain, title=title),
            },
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for domain, title in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-written text."""

    # Preprocess data
    df, prompts = process_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI-written text based on domain and title."
    )
    parser.add_argument("llm_name", type=str, help="Name of the LLM model")
    parser.add_argument("llm_path", type=str, help="Path to the LLM model")
    parser.add_argument(
        "quant",
        type=str,
        nargs="?",
        default=None,
        help="Quantization setting (optional)",
    )

    args = parser.parse_args()

    main(args.llm_name, args.llm_path, args.quant)
