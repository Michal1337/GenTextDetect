import csv
import os
from typing import List, Tuple

import pandas as pd
import tiktoken
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from params import DATA_AI_PATH, DATA_HUMAN_PATH, MASTER_STATS_PATH, STATS_PATH
from utils import get_csv_paths


def calc_stats(
    texts: List[str], tokenizer: tiktoken
) -> Tuple[List[List[int]], int, int, int, int, int]:
    results = []
    total_sentences, total_words, total_chars, total_tokens = 0, 0, 0, 0
    total_samples = len(texts)

    for text in tqdm(texts):
        text_words = 0
        text_chars = 0
        text_tokens = tokenizer.encode(text)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenize(sentence)
            text_words += len(words)
            text_chars += sum([len(word) for word in words])

        total_sentences += len(sentences)
        total_words += text_words
        total_chars += text_chars
        total_tokens += len(text_tokens)

        results.append([len(sentences), text_words, text_chars, len(text_tokens)])
    return (
        results,
        total_samples,
        total_sentences,
        total_words,
        total_chars,
        total_tokens,
    )


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("o200k_base")  # cl100k_base
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)

    with open(MASTER_STATS_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "data",
                "model",
                "num_samples",
                "num_sentences",
                "num_words",
                "num_chars",
                "num_tokens",
            ]
        )

    for path in paths:
        print(f"Processing {path}...")
        if path.split("_")[-1] == "human.csv":
            stats_path = os.path.join(
                STATS_PATH,
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_stats.csv"),
            )
        else:
            stats_path = os.path.join(
                STATS_PATH,
                path.split("/")[-3],
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_stats.csv"),
            )

        df = pd.read_csv(path)
        texts = df["text"].values

        results, num_samples, num_sentences, num_words, num_chars, num_tokens = (
            calc_stats(texts, tokenizer)
        )

        results = pd.DataFrame(
            results, columns=["num_sentences", "num_words", "num_chars", "num_tokens"]
        )
        results.to_csv(stats_path, index=False)

        data_name, model = path.split("/")[-1].split("_")
        model = model.removesuffix(".csv")

        with open(MASTER_STATS_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    data_name,
                    model,
                    num_samples,
                    num_sentences,
                    num_words,
                    num_chars,
                    num_tokens,
                ]
            )
