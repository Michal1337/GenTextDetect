import argparse
import random
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from gen_params import (
    AI_DATA_BASE_PATH,
    HUMAN_DATA_BASE_PATH,
    MAX_TOKENS_PROMPT,
    RAW_DATA_BASE_PATH,
    SAMPLING_PARAMS,
    SEED,
)
from gen_utils import check_for_too_long_prompts, generate_texts

np.random.seed(SEED)
random.seed(SEED)

DS_NAME = "google-research-datasets/natural_questions"
RAW_DATA_PATH = RAW_DATA_BASE_PATH + "natural_questions.csv"
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "natural-questions_human.csv"
AI_DATA_PATH = AI_DATA_BASE_PATH + "natural_questions/natural-questions_"

PROMPT_COLS = ["document", "question"]
TEXT_COL = "answer"
TO_DROP = ["document", "question"]

BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for answering questions based on the provided context. "
        "The context will be a copy of a Wikipedia article. Answer the question based only on the given context. "
        "MAKE SURE TO REPLY ONLY WITH THE ANSWER.",
    },
    {"role": "user", "content": "Context:\n{context}\nQuestion: {question}"},
    {"role": "assistant", "content": "Answer:\n"},
]

PERCENT_SAMPLE = 0.05
BATCH_SIZE = 32


def nq2csv(dataset, save_path, batch_size):
    """Convert the dataset to CSV format with batched writing."""
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["document", "question", "answer"])

    for split in ["train", "validation"]:
        documents, questions, answers = [], [], []

        for item in tqdm(dataset[split], desc=f"Processing {split} split"):
            idx = np.random.randint(len(item["long_answer_candidates"]["start_token"]))
            start = item["long_answer_candidates"]["start_token"][idx]
            end = item["long_answer_candidates"]["end_token"][idx]
            tokens = item["document"]["tokens"]

            question = " ".join(token for token in item["question"]["tokens"])

            ans = tokens["token"][start:end]
            ans_is_html = tokens["is_html"][start:end]
            ans = " ".join([token for token, html in zip(ans, ans_is_html) if not html])

            doc_is_html = tokens["is_html"]
            document = " ".join(
                [token for token, html in zip(tokens["token"], doc_is_html) if not html]
            )

            documents.append(document)
            questions.append(question)
            answers.append(ans)

            if len(documents) == batch_size:
                with open(save_path, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerows(zip(documents, questions, answers))

                documents, questions, answers = [], [], []

        # Save the remaining batch
        with open(save_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(zip(documents, questions, answers))


def preprocess_data() -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Preprocess the Natural Questions dataset and create prompts."""

    # Load dataset and convert to CSV
    # dataset = load_dataset(DS_NAME)
    # nq2csv(dataset, RAW_DATA_PATH, BATCH_SIZE)

    # Read the processed CSV
    df = pd.read_csv(RAW_DATA_PATH)
    df.dropna(inplace=True)
    df.drop_duplicates(subset="answer", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create prompts
    prompts = [
        [
            BASE_PROMPT[0],
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    context=context, question=question
                ),
            },
            BASE_PROMPT[2],
        ]
        for context, question in df[PROMPT_COLS].values
    ]

    # Remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    return df, prompts


def main(llm_name: str, llm_path: str, quant: Optional[str] = None) -> None:
    """Main function to preprocess data and generate AI responses."""

    # Preprocess data
    df, prompts = preprocess_data()

    prompts = random.sample(prompts, int(len(prompts) * PERCENT_SAMPLE))

    # Generate AI responses
    generate_texts(
        prompts, llm_name, llm_path, quant, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH
    )


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Generate AI responses for Natural Questions dataset."
    )
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
