import argparse
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, SAMPLING_PARAMS, SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

DS_NAME = "euclaise/writingprompts"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "writingprompts_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "writingprompts/writingprompts_"  # Path to save the generated data

PROMPT_COLS = ["prompt"]  # Columns with the prompt data
TEXT_COL = "story"  # Column with the text data
TO_DROP = ["prompt", "prompt_length"]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing stories based on the provided prompt. Based on the given prompt, generate a story that aligns with it. MAKE SURE TO REPLY ONLY WITH THE STORY.",
    },
    {"role": "user", "content": "Prompt:\n{prompt}"},
    {"role": "assistant", "content": "Story:\n"},
]

PERCENT_SAMPLE = 0.05
BATCH_SIZE = 128  # Number of prompts to generate at once


def remove_prefix(text: str, pattern: str) -> str:
    """Remove the prefix from the prompt."""
    return re.sub(pattern, "", text)


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the writing prompts dataset."""
    dataset = load_dataset(DS_NAME)
    df = pd.concat(
        [
            dataset["train"].to_pandas(),
            dataset["validation"].to_pandas(),
            dataset["test"].to_pandas(),
        ]
    )

    # Find and remove common prefixes in the prompt column
    prefixes = [prompt[:6] for prompt in df["prompt"].values]
    unique, counts = np.unique(prefixes, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique, counts = unique[sorted_indices], counts[sorted_indices]
    prefixes = unique[:14]
    pattern = r"^(?:" + "|".join(re.escape(prefix) for prefix in prefixes) + r")\s*"
    df["prompt"] = df["prompt"].apply(remove_prefix, pattern=pattern)

    # Filter and clean data
    df["prompt_length"] = df["prompt"].str.len()
    df = df[df["prompt_length"] > 0]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(prompt=prompt),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for prompt in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-written stories based on the prompts."""

    # Preprocess data
    df, prompts = process_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate AI-written Stories based on prompts."
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

    # Call the main function with the parsed arguments
    main(args.llm_name, args.llm_path, args.quant)
