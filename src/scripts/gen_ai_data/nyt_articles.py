import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from gen_params import (AI_DATA_BASE_PATH, HUMAN_DATA_BASE_PATH,
                        MAX_TOKENS_PROMPT, RAW_DATA_BASE_PATH, SAMPLING_PARAMS)
from gen_utils import check_for_too_long_prompts, generate_texts

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "nyt-articles-2020.csv"
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "nyt-articles_human.csv"
AI_DATA_PATH = AI_DATA_BASE_PATH + "nyt_articles/nyt-articles_"

PROMPT_COLS = ["headline", "keywords"]
TEXT_COL = "abstract"
TO_DROP = [
    "newsdesk",
    "section",
    "subsection",
    "material",
    "headline",
    "keywords",
    "word_count",
    "pub_date",
    "n_comments",
    "uniqueID",
    "length_abstract",
    "length_headline",
]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing article abstracts. "
        "Based on the provided headline and list of keywords, generate an abstract for the article. "
        "Ensure the abstract maintains a similar length to typical article abstracts. "
        "MAKE SURE TO REPLY ONLY WITH THE ABSTRACT.",
    },
    {"role": "user", "content": "Headline:\n{headline}\nKeywords:\n{keywords}"},
    {"role": "assistant", "content": "Abstract:\n"},
]

BATCH_SIZE = 64


def process_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Load and preprocess the dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    df.dropna(subset=[TEXT_COL], inplace=True)

    df["length_abstract"] = df[TEXT_COL].str.len()
    df["length_headline"] = df["headline"].str.len()

    df = df[(df["length_abstract"] >= 50) & (df["length_headline"] >= 10)]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare prompts
    prompts = []
    for headline, keywords in df[PROMPT_COLS].values:
        try:
            kw = ", ".join(eval(keywords))  # Convert string list to actual list
        except TypeError:
            kw = "None"

        prompt = [
            BASE_PROMPT[0],
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    headline=headline, keywords=kw
                ),
            },
            BASE_PROMPT[2],
        ]
        prompts.append(prompt)

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to generate AI-generated abstracts."""

    # Preprocess data
    df, prompts = process_data()

    # Generate AI data
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI-written article abstracts for NYT articles."
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
