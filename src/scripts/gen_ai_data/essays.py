import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS)
from gen_utils import check_for_too_long_prompts, generate_texts

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "essays.csv"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "essays_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "essays/essays_"  # Path to save the generated data

PROMPT_COLS = ["text"]
TEXT_COL = "text"
TO_DROP = ["essay_id", "label", "source", "prompt", "text_length"]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for rewriting students' essays. "
        "Based on the provided essay, generate a similar one in a natural "
        "and authentic tone, maintaining the same meaning but rephrased. "
        "Ensure the rewritten essay matches the length of the original, "
        "and avoids overly formal or advanced phrasing. "
        "MAKE SURE TO REPLY ONLY WITH THE SIMILAR ESSAY.",
    },
    {"role": "user", "content": "Essay: \n {essay}"},
    {"role": "assistant", "content": "Similar essay: \n"},
]

BATCH_SIZE = 32  # Number of prompts to generate at once


def preprocess_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Preprocess the essays dataset and prepare the prompts."""

    # Load and preprocess data
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["label"] == 0]  # Filter out rows with label not equal to 0
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]  # Filter out essays that are too short
    df.drop_duplicates(subset=TEXT_COL, inplace=True)  # Remove duplicates
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts for generation
    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(essay=essay),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for essay in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-rewritten essays."""

    # Preprocess data
    df, prompts = preprocess_data()

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Generate AI-rewritten essays.")
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
