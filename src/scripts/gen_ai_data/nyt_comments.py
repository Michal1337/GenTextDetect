import argparse
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS, SEED)
from gen_utils import check_for_too_long_prompts, generate_texts

random.seed(SEED)

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "nyt-comments-2020.csv"
ARTICLES_PATH = RAW_DATA_BASE_PATH + "nyt-articles-2020.csv"  # Path to article abstracts

HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "nyt-comments_human.csv"
AI_DATA_PATH = AI_DATA_BASE_PATH + "nyt_comments/nyt-comments_"

PROMPT_COLS = ["abstract", "commentBody"]
TEXT_COL = "commentBody"
TO_DROP = [
    "articleID",
    "abstract",
    "length_comment",
    "length_abstract",
]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing comments based on article abstracts and sample comments. "
        "Based on the provided article abstract and sample comment, generate a similar comment related to the article. "
        "Ensure the comment matches the tone and length of the sample comment. "
        "MAKE SURE TO REPLY ONLY WITH THE COMMENT.",
    },
    {"role": "user", "content": "Abstract:\n{abstract}\nComment:\n{comment}"},
    {"role": "assistant", "content": "Similar comment:\n"},
]

PERCENT_SAMPLE = 0.05
BATCH_SIZE = 512


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the dataset."""
    df = pd.read_csv(RAW_DATA_PATH, usecols=["commentBody", "articleID"])
    df_articles = pd.read_csv(ARTICLES_PATH, usecols=["abstract", "uniqueID"])

    # Merge article abstracts with comments
    df = df.join(df_articles.set_index("uniqueID"), on="articleID")
    df.dropna(inplace=True)

    # Filter comments and abstracts by length
    df["length_comment"] = df[TEXT_COL].str.len()
    df["length_abstract"] = df["abstract"].str.len()
    df = df[(df["length_comment"] >= 50) & (df["length_abstract"] >= 50)]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts
    prompts = [
        [
            BASE_PROMPT[0],
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    abstract=abstract, comment=comment
                ),
            },
            BASE_PROMPT[2],
        ]
        for abstract, comment in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-written comments."""

    # Preprocess data
    df, prompts = process_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI-written comments for NYT articles."
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
