import pandas as pd
from datasets import load_dataset

from gen_params import *
from gen_utils import *

DS_NAME = "liamdugan/raid"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "raid_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "raid/raid_"  # Path to save the generated data

PROMPT_COLS = ["domain", "title"]  # Columns with the prompt data
TEXT_COL = "generation"  # Column with the text data
TO_DROP = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation_length",
    "title_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant specializing in writing texts across various domains, including abstracts and news articles, based on provided titles. Based on provided domain and title generate a text of appropriate length related to the domain and title. MAKE SURE TO REPLAY ONLY WITH THE GENERATED TEXT.",
    },
    {"role": "user", "content": "Domain: \n {domain} \n Title: {title}"},
    {"role": "assistant", "content": "Generated text: \n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


if __name__ == "__main__":
    dataset = load_dataset(DS_NAME, "raid")["train"]
    dataset = dataset.filter(lambda x: x["model"] == "human")
    df = dataset.to_pandas()
    df["title_length"] = df["title"].str.len()
    df["generation_length"] = df[TEXT_COL].str.len()
    df = df[(df["title_length"] >= 10) & (df["generation_length"] >= 50)]
    df.drop_duplicates(subset=[TEXT_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(domain=domain, title=title),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for domain, title in df[PROMPT_COLS].values
    ]
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
