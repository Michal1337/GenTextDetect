import csv
import os
from collections import Counter
from typing import Dict, List

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tqdm import tqdm

from params import (DATA_AI_PATH, DATA_HUMAN_PATH, MAX_NGRAM_LEVEL,
                    MIN_NGRAM_LEVEL, NGRAMS_PATH)
from utils import get_csv_paths


def calc_ngrams(texts: List[str], min_n: int, max_n: int) -> Dict[int, Counter]:
    ngrams_frequencies = {}

    for text in tqdm(texts):
        tokens = word_tokenize(text.lower())

        words_only = [token for token in tokens if token.isalpha()]

        for n in range(min_n, max_n + 1):
            if n not in ngrams_frequencies:
                ngrams_frequencies[n] = Counter()
            ngrams_generated = ngrams(words_only, n)
            ngrams_frequencies[n].update(ngrams_generated)

    return ngrams_frequencies


def save_ngrams_to_csv(
    ngrams_frequencies: Dict[int, Counter], csv_filename: str
) -> None:
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["n", "ngram", "frequency"])

        for n, counter in ngrams_frequencies.items():
            sorted_ngrams = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            for ngram, count in sorted_ngrams:
                writer.writerow([n, " ".join(ngram), count])


if __name__ == "__main__":
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)
    for path in paths:
        if path.split("_")[-1] == "human.csv":
            ngrams_path = os.path.join(
                NGRAMS_PATH,
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_ngrams.csv"),
            )
        else:
            ngrams_path = os.path.join(
                NGRAMS_PATH,
                path.split("/")[-3],
                path.split("/")[-2],
                path.split("/")[-1].replace(".csv", "_ngrams.csv"),
            )

        df = pd.read_csv(path)
        texts = df["text"].values
        ngrams_frequencies = calc_ngrams(texts, MIN_NGRAM_LEVEL, MAX_NGRAM_LEVEL)

        save_ngrams_to_csv(ngrams_frequencies, ngrams_path)
