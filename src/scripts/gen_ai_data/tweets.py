import argparse
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS)
from gen_utils import check_for_too_long_prompts, generate_texts

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "tweets.csv"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "tweets_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "tweets/tweets_"  # Path to save the generated data

PROMPT_COLS = ["text"]  # Columns with the prompt data
TEXT_COL = "text"  # Column with the text data
TO_DROP = [
    "target",
    "ids",
    "date",
    "flag",
    "user",
    "text_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for rewriting tweets. Based on the provided tweet, generate a similar one while maintaining the original meaning and tone. MAKE SURE TO REPLY ONLY WITH THE SIMILAR TWEET.",
    },
    {"role": "user", "content": "Tweet:\n{tweet}"},
    {"role": "assistant", "content": "Similar tweet:\n"},
]
PERCENT_SAMPLE = 0.4
BATCH_SIZE = 512  # Number of prompts to generate at once


def standard_chars(s: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9\s.,!?\'\"-]+$", s))


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the tweets dataset."""
    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin-1",
        names=["target", "ids", "date", "flag", "user", "text"],
    )

    # Filter tweets with only standard characters
    df = df[df[TEXT_COL].apply(standard_chars)]

    # Filter out tweets that are too short
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]

    # Remove duplicates and reset index
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(tweet=tweet),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for tweet in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-written tweets."""

    # Preprocess data
    df, prompts = process_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI-written Tweets.")
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
