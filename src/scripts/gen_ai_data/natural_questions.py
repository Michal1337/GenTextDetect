import numpy as np
import pandas as pd
from datasets import load_dataset

from gen_params import *
from gen_utils import *

np.random.seed(SEED)

# DS_NAME = "google-research-datasets/natural_questions"  # Path to the raw data
RAW_DATA_PATH = RAW_DATA_BASE_PATH + "natural_questions.csv" # Path to save the raw data
HUMAN_DATA_PATH = HUMAN_DATA_BASE_PATH + "natural_questions_human.csv" # Path to the human data
AI_DATA_PATH = AI_DATA_BASE_PATH + "natural_questions/natural-questions_" # Path to save the generated data

PROMPT_COLS = ["document", "question"]  # Columns with the prompt data
TEXT_COL = "answer"  # Column with the text data
TO_DROP = ["document", "question"]  # Columns to drop from the human data
BASE_PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for answering questions based on the provided context. The context will be a copy of a Wikipedia article. Answer the question based only on the given context. MAKE SURE TO REPLY ONLY WITH THE ANSWER.",
    },
    {"role": "user", "content": "Context:\n{context}\nQuestion: {question}"},
    {"role": "assistant", "content": "Answer:\n"},
]

BATCH_SIZE = 128  # Number of prompts to generate at once


def nq2csv(dataset, save_path, batch_size):
    # init csv
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["document", "question", "answer"])

    for split in ["train", "validation"]:
        documents, questions, answers = [], [], []

        for item in tqdm(dataset[split]):
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
                    for document, question, answer in zip(
                        documents, questions, answers
                    ):
                        writer.writerow([document, question, answer])

                documents, questions, answers = [], [], []

        # save last, not full batch
        with open(save_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for document, question, answer in zip(documents, questions, answers):
                writer.writerow([document, question, answer])


if __name__ == "__main__":
    dataset = load_dataset(DS_NAME)
    nq2csv(dataset, RAW_DATA_PATH, BATCH_SIZE)

    df = pd.read_csv(RAW_DATA_PATH)

    prompts = [
        [
            BASE_PROMPT[0],  # The system message
            {
                "role": "user",
                "content": BASE_PROMPT[1]["content"].format(
                    context=context, question=question
                ),
            },  # Formatted user message
            BASE_PROMPT[2],  # Start of the assistant message
        ]
        for context, question in df[PROMPT_COLS].values
    ]
    # remove too long prompts
    df, prompts = check_for_too_long_prompts(df, prompts, MAX_TOKENS_PROMPT)

    # Save human data
    df.drop(TO_DROP, axis=1, inplace=True)
    df.rename(columns={TEXT_COL: "text"}, inplace=True)
    df.to_csv(HUMAN_DATA_PATH, index=False)

    # Generate AI data
    generate_texts(prompts, LLMS, SAMPLING_PARAMS, BATCH_SIZE, AI_DATA_PATH)
