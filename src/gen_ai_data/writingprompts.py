import re

import numpy as np
import pandas as pd
from datasets import load_dataset

from gen_params import *
from gen_utils import *

DS_NAME = "euclaise/writingprompts"  # Path to the raw data
HUMAN_DATA_PATH = "../../data/data_human/writingprompts_human.csv" # Path to the human data
AI_DATA_PATH = "../../data/data_ai/writingprompts/writingprompts_"  # Path to save the generated data

PROMPT_COLS = ["prompt"]  # Columns with the prompt data
TEXT_COL = "story"  # Column with the text data
TO_DROP = ["prompt", "prompt_length"]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing stories based on provided prompt. Based on provided prompt generate a story. MAKE SURE TO REPLAY ONLY WITH THE STORY.",
    },
    {"role": "user", "content": "Prompt: \n {prompt}"},
    {"role": "assistant", "content": "Story: \n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


def remove_prefix(text: str) -> str:
    """Remove the prefix from the prompt."""
    return re.sub(pattern, "", text)


if __name__ == "__main__":
    dataset = load_dataset(DS_NAME)
    df = pd.concat(
        [
            dataset["train"].to_pandas(),
            dataset["validation"].to_pandas(),
            dataset["test"].to_pandas(),
        ]
    )

    prefixes = [prompt[:6] for prompt in df["prompt"].values]
    unique, counts = np.unique(prefixes, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique, counts = unique[sorted_indices], counts[sorted_indices]
    prefixes = unique[:14]
    pattern = r"^(?:" + "|".join(re.escape(prefix) for prefix in prefixes) + r")\s*"
    df["prompt"] = df["prompt"].apply(remove_prefix)
    df["prompt_length"] = df["prompt"].str.len()
    df = df[df["prompt_length"] > 0]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)


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
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
