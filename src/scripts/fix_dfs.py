import re

import pandas as pd
import tiktoken
from tqdm import tqdm

from params import DATA_AI_PATH, DATA_HUMAN_PATH
from utils import get_csv_paths


def clean_text(s: str) -> str:
    """
    Remove a leading '", [' or '[' and a trailing '", ]' or ']' from the given string.
    """
    # Remove leading patterns
    try:
        for prefix in ('"', "[", '["'):
            if s.startswith(prefix):
                s = s[len(prefix) :]
                break

        # Remove trailing patterns
        for suffix in ('"', "]", '"]'):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break

        s = s.replace("  ", "")
        s = s.strip()
        s = re.sub(r"([^\w\s])\1{2,}", r"\1\1", s)

        s = re.sub(r"([\n\t])\1{2,}", r"\1\1", s)
    except (AttributeError, TypeError):
        pass

    return s


def remove_errors(path: str, tokenizer: tiktoken) -> None:
    print(f"Processing {path}...")
    df = pd.read_csv(path)
    df["text"] = df["text"].apply(clean_text)
    texts = df["text"].tolist()

    print(f"Number of texts: {len(texts)}")

    err = []
    for i, text in enumerate(tqdm(texts)):
        try:
            tokenizer.encode(text)
        except TypeError:
            err.append([i, text])

    if len(err) > 0:
        print(f"{len(err)} Errors in {path}:")
        for i, text in err:
            print(f"Index: {i}, Text: {text}")

        user_input = input(f"Do you want to remove the errors in {path}? (y/n): ")
        if user_input.lower() == "y":
            df.drop(index=[i for i, _ in err], inplace=True)
        else:
            print(f"Errors in {path} were not removed.")

    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("o200k_base")  # cl100k_base
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)

    for path in paths:
        remove_errors(path, tokenizer)
