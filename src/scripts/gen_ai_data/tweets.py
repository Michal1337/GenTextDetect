import re
import pandas as pd

from gen_params import *
from gen_utils import *

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "tweets.csv"  # Path to the raw data
HUMAN_DATA_PATH = "../../data/data_human/tweets_human.csv"  # Path to the human data
AI_DATA_PATH = "../../data/data_ai/tweets/tweets_"  # Path to save the generated data

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
BATCH_SIZE = 8  # Number of prompts to generate at once


def standard_chars(s: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9\s.,!?\'\"-]+$", s))


if __name__ == "__main__":
    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin-1",
        names=["target", "ids", "date", "flag", "user", "text"],
    )
    df = df[df[TEXT_COL].apply(standard_chars)]
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 15]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

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
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
