import pandas as pd

from gen_params import *
from gen_utils import *

RAW_DATA_PATH = "../../data/data_raw/blogs.csv"  # Path to the raw data
HUMAN_DATA_PATH = "../../data/data_human/blogs_human.csv"  # Path to the human data
AI_DATA_PATH = "../../data/data_ai/blogs/blogs_"  # Path to save the generated data

PROMPT_COLS = ["text"]  # Columns with the prompt data
TEXT_COL = "text"  # Column with the text data
TO_DROP = [
    "id",
    "gender",
    "age",
    "topic",
    "sign",
    "date",
    "text_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful asistant for rewritting blogs. Based on provided blog generate a similar one. MAKE SURE TO REPLAY ONLY WITH THE SIMILAR BLOG.",
    },
    {"role": "user", "content": "Blog: \n {blog} \n"},
    {"role": "assistant", "content": "Similar blog: \n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

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
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
