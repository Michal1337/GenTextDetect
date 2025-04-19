import argparse
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, SAMPLING_PARAMS, SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

DS_NAME = "EdinburghNLP/xsum"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "xsum_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "xsum/xsum_"  # Path to save the generated data

PROMPT_COLS = ["summary"]  # Columns with the prompt data
TEXT_COL = "document"  # Column with the text data
TO_DROP = [
    "summary",
    "id",
    "summary_length",
    "document_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing news articles based on provided one-sentence summaries. Based on the given summary, generate a full news article. MAKE SURE TO REPLY ONLY WITH THE NEWS ARTICLE.",
    },
    {"role": "user", "content": "Summary:\n{summary}"},
    {"role": "assistant", "content": "News article:\n"},
]
PERCENT_SAMPLE = 0.15
BATCH_SIZE = 256  # Number of prompts to generate at once


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the xsum dataset."""
    dataset = load_dataset(DS_NAME)
    df = pd.concat(
        [
            dataset["train"].to_pandas(),
            dataset["validation"].to_pandas(),
            dataset["test"].to_pandas(),
        ]
    )

    df["summary_length"] = df["summary"].str.len()
    df["document_length"] = df[TEXT_COL].str.len()
    df = df[(df["summary_length"] >= 10) & (df["document_length"] >= 50)]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(summary=summary),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for summary in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-written news articles based on the prompts."""

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
        description="Generate AI-written news articles based on summaries."
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
