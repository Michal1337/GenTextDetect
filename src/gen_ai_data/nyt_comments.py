import pandas as pd

from gen_params import *
from gen_utils import *

RAW_DATA_PATH = "../../data/data_raw/nyt-comments-2020.csv"  # Path to the raw data
HUMAN_DATA_PATH = "../../data/data_human/nyt_comments_human.csv"  # Path to the human data
AI_DATA_PATH = "../../data/data_ai/nyt_comments/nyt_comments_"  # Path to save the generated data


PROMPT_COLS = ["abstract", "commentBody"]  # Columns with the prompt data
TEXT_COL = "commentBody"  # Column with the text data
TO_DROP = [
    "articleID",
    "abstract",
    "length_comment",
    "length_abstract",
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful asistant for writing comments based on article abstracts and sample comments. Based on provided article abstract and a sample comment generate similar comment related to the article. MAKE SURE TO REPLAY ONLY WITH THE COMMENT.",
    },
    {"role": "user", "content": "Abstract: \n {abstract} \n  Comment: \n {comment}."},
    {"role": "assistant", "content": "Similar comment: \n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, usecols=["commentBody", "articleID"])
    df_articles = pd.read_csv(
        "../../data/data_raw/nyt-articles-2020.csv", usecols=["abstract", "uniqueID"]
    )
    df = df.join(df_articles.set_index("uniqueID"), on="articleID")
    df.dropna(inplace=True)
    df["length_comment"] = df[TEXT_COL].str.len()
    df["length_abstract"] = df["abstract"].str.len()
    df = df[(df["length_comment"] >= 50) & (df["length_abstract"] >= 50)]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    abstract=abstract, comment=comment
                ),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for abstract, comment in df[PROMPT_COLS].values
    ]
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
