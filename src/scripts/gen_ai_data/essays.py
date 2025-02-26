import pandas as pd

from gen_params import *
from gen_utils import *

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "essays.csv"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "essays_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "essays/essays_"  # Path to save the generated data

PROMPT_COLS = ["text"]  # Columns with the prompt data
TEXT_COL = "text"  # Column with the text data
TO_DROP = [
    "essay_id",
    "label",
    "source",
    "prompt",
    "text_length",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for rewriting students' essays. Based on the provided essay, generate a similar one in a natural and authentic tone, maintaining the same meaning but rephrased. Ensure the rewritten essay matches the length of the original, and avoids overly formal or advanced phrasing. MAKE SURE TO REPLY ONLY WITH THE SIMILAR ESSAY.",
    },
    {"role": "user", "content": "Essay: \n {essay}"},
    {"role": "assistant", "content": "Similar essay: \n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["label"] == 0]
    df["text_length"] = df[TEXT_COL].str.len()
    df = df[df["text_length"] >= 50]
    df.drop_duplicates(TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

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
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
