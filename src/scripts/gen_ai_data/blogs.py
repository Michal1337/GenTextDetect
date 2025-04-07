import argparse
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS,
                        SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "blogs.csv"
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "blogs_human.csv"
AI_DATA_PATH = AI_DATA_BASE_PATH + "blogs/blogs_"

PROMPT_COLS = ["text"]
TEXT_COL = "text"
TO_DROP = ["id", "gender", "age", "topic", "sign", "date", "text_length"]
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful asistant for rewritting blogs. Based on provided blog generate a similar one. MAKE SURE TO REPLAY ONLY WITH THE SIMILAR BLOG.",
    },
    {"role": "user", "content": "Blog:\n{blog}"},
    {"role": "assistant", "content": "Similar blog:\n"},
]

PERCENT_SAMPLE = 0.05
BATCH_SIZE = 64


def preprocess_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Preprocess the blogs dataset and prepare the prompts."""

    # Load and preprocess data
    df = pd.read_csv(RAW_DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(blog=blog),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for blog in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-rewritten blogs."""

    # Preprocess data
    df, prompts = preprocess_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Generate AI-rewritten blogs.")
    parser.add_argument("llm_name", type=str, help="Name of the LLM model")
    parser.add_argument("llm_path", type=str, help="Path to the LLM model")
    parser.add_argument(
        "quant",
        type=str,
        nargs="?",
        default=None,
        help="Quantization setting (can be None)",
    )

    args = parser.parse_args()

    main(args.llm_name, args.llm_path, args.quant)
