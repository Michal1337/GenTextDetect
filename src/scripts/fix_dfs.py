import pandas as pd
import tiktoken

from params import DATA_AI_PATH, DATA_HUMAN_PATH
from utils import get_csv_paths


def fix_text(text: str) -> str:
    if isinstance(eval(text), list) and len(eval(text)) == 1:
        text = eval(text)[0]
    return text


def remove_errors(path: str) -> None:
    print(f"Processing {path}...")
    df = pd.read_csv(path)
    texts = df["text"].tolist()

    print(f"Number of texts: {len(texts)}")

    err = []
    for i, text in enumerate(texts):
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
            # df["text"] = df["text"].apply(fix_text)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(path, index=False)
        else:
            print(f"Errors in {path} were not removed.")


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("o200k_base")  # cl100k_base
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)

    for path in paths:
        remove_errors(path)
