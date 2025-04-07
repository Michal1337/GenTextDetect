import argparse
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS,
                        SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "reddit.csv"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "reddit_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "reddit/reddit_"  # Path to save the generated data

PROMPT_COLS = ["body", "subreddit"]  # Columns with the prompt data
TEXT_COL = "body"  # Column with the text data
TO_DROP = [
    "subreddit",
    "controversiality",
    "score",
    "text_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for rewriting Reddit comments. Based on the provided comment and the subreddit where it was posted, generate a similar comment that fits the context of the subreddit. MAKE SURE TO REPLY ONLY WITH THE SIMILAR COMMENT.",
    },
    {"role": "user", "content": "Comment:\n{comment}\nSubreddit:\n{subreddit}"},
    {"role": "assistant", "content": "Similar comment:\n"},
]

PERCENT_SAMPLE = 0.2
BATCH_SIZE = 256  # Number of prompts to generate at once


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the Reddit dataset."""
    df = pd.read_csv(RAW_DATA_PATH)

    # Filter out comments that are too short
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]

    # Remove unwanted subreddits
    df = df[df["subreddit"] != "Pikabu"]

    # Remove duplicates and reset index
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    comment=comment, subreddit=subreddit
                ),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for comment, subreddit in df[PROMPT_COLS].values
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
    parser = argparse.ArgumentParser(description="Generate AI-written Reddit comments.")
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
