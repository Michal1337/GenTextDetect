import os
import numpy as np
import pandas as pd
from typing import List, Dict


def get_csv_paths(folder_path: str, recursive: bool = False) -> List[str]:
    if recursive:
        # Walk through all subdirectories
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if file.endswith(".csv")
        ]
    else:
        # Get files in the root folder only
        file_paths = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".csv")
        ]

    return file_paths


def create_dataset_idx(
    max_tokens: int,
    batch_size: int,
    stats: Dict[str, pd.DataFrame],
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    p0: float,
    p1: float,
    save_path: str,
):
    total_tokens = 0
    total_sentences = 0
    total_samples = 0
    prob = p1 / (p0 + p1)

    while total_tokens < max_tokens:
        p = np.random.rand()
        if p < prob:
            slct = df0.sample(n=1, weights=df0["prob"].values)
        else:
            slct = df1.sample(n=1, weights=df1["prob"].values)

        data, model = slct["data"].values[0], slct["model"].values[0]
        stat = stats[f"{data}_{model}"]

        if len(stat) < batch_size:
            continue
        
        # select batch_size random rows from stat and remove them
        slct = stat.sample(n=batch_size)
        stat.drop(slct.index, inplace=True)

        total_tokens += slct.sum()["num_tokens"]
        total_sentences += slct.sum()["num_sentences"]
        total_samples += batch_size

        # save data, model, slct.index to csv
        slct["data"] = data
        slct["model"] = model
        slct.reset_index(inplace=True)
        slct.drop(
            columns=["num_sentences", "num_words", "num_chars", "num_tokens"],
            inplace=True,
        )
        slct.to_csv(
            save_path, mode="a", header=not os.path.exists(save_path), index=False
        )

        if total_samples % 1000 == 0:
            print(
                f"Total samples: {total_samples}, Total sentences: {total_sentences}, Total tokens: {total_tokens}"
            )

    print(
        f"Final samples: {total_samples}, Final sentences: {total_sentences}, Final tokens: {total_tokens}"
    )
