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
    df_main: pd.DataFrame,
    col_c0: str,
    save_path: str,
):

    for ds in df_main["data"].unique():
        mask_c1 = (df_main["data"].values == ds) & (df_main["model"].values != col_c0)
        mask_c0 = (df_main["data"].values == ds) & (df_main["model"].values == col_c0)

        df_main.loc[mask_c1, "prob"] = (
            1
            / df_main.loc[mask_c1, "avg_sent_per_sample"]
            / (1 / df_main.loc[mask_c1, "avg_sent_per_sample"]).sum()
        )

        avg_c0 = df_main.loc[mask_c0, "avg_sent_per_sample"].values[0]
        avg_c1 = (
            df_main.loc[mask_c1, "avg_sent_per_sample"] * df_main.loc[mask_c1, "prob"]
        ).sum()

        c = 1 / (1 + avg_c1 / avg_c0)
        p = 1 - c

        df_main.loc[mask_c1, "prob"] *= c
        df_main.loc[mask_c0, "prob"] = p

    weights = [
        1
        / (
            df_main.loc[df_main["data"] == ds, "avg_sent_per_sample"]
            * df_main.loc[df_main["data"] == ds, "prob"]
        ).sum()
        for ds in df_main["data"].unique()
    ]
    probs = np.array(weights) / np.sum(weights)

    total_tokens = 0
    total_sentences = 0
    total_samples = 0

    while total_tokens < max_tokens:
        data = np.random.choice(df_main["data"].unique(), p=probs)
        tmp = df_main[(df_main["data"] == data)]
        model = np.random.choice(tmp["model"], p=tmp["prob"])

        stat = stats[f"{data}_{model}"]

        slct = stat.sample(n=batch_size)
        stat.drop(slct.index, inplace=True)

        total_tokens += slct.sum()["num_tokens"]
        total_sentences += slct.sum()["num_sentences"]
        total_samples += batch_size

        # save data, model, slct.index to csv
        slct["data"] = data
        slct["model"] = model
        slct.reset_index(inplace=True)
        # slct.drop(columns=["num_sentences", "num_words", "num_chars", "num_tokens"], inplace=True)
        slct.to_csv(
            save_path, mode="a", header=not os.path.exists(save_path), index=False
        )

        cnt += 1
        if cnt % 1000 == 0:
            print(
                f"total_tokens: {total_tokens}, total_sentences: {total_sentences}, total_samples: {total_samples}"
            )

    print(
        f"Final samples: {total_samples}, Final sentences: {total_sentences}, Final tokens: {total_tokens}"
    )
