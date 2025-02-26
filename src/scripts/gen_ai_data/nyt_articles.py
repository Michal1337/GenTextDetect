import pandas as pd

from gen_params import *
from gen_utils import *

RAW_DATA_PATH = RAW_DATA_BASE_PATH + "nyt-articles-2020.csv"  # Path to the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "nyt_articles_human.csv"  # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "nyt_articles/nyt-articles_"  # Path to save the generated data


PROMPT_COLS = ["headline", "keywords"]  # Columns with the prompt data
TEXT_COL = "abstract"  # Column with the text data
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
]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for writing article abstracts. Based on the provided headline and list of keywords, generate an abstract for the article. Ensure the abstract maintains a similar length to typical article abstracts. MAKE SURE TO REPLY ONLY WITH THE ABSTRACT.",
    },
    {"role": "user", "content": "Headline:\n{headline}\nKeywords:\n{keywords}"},
    {"role": "assistant", "content": "Abstract:\n"},
]
BATCH_SIZE = 8  # Number of prompts to generate at once


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    df.dropna(subset=[TEXT_COL], inplace=True)
    df["length_abstract"] = df[TEXT_COL].str.len()
    df["length_headline"] = df["headline"].str.len()

    df = df[(df["length_abstract"] >= 50) & (df["length_headline"] >= 10)]
    df.drop_duplicates(subset=TEXT_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    prompts = []
    for headline, keywords in df[PROMPT_COLS].values:
        try:
            kw = ", ".join(eval(keywords))
        except:
            kw = "None"
        prompt = [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    headline=headline, keywords=kw
                ),
            },  # Formatted user message
            BASE_PROMPT[2],  # The assistant message
        ]
        prompts.append(prompt)

    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
