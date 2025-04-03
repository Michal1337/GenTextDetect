import csv
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    cols_c0: List[str],
    save_path: str,
):

    for ds in df_main["data"].unique():
        mask_c0 = (df_main["data"].values == ds) & (df_main["model"].isin(cols_c0))
        mask_c1 = (df_main["data"].values == ds) & (~df_main["model"].isin(cols_c0))

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


def idx2csv(df: pd.Dataframe, cols_c0: List[str], save_path: str) -> None:
    # init csv
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

    # iterate through every data and model combination
    for data in tqdm(df["data"].unique()):
        for model in df["model"].unique():
            if model == "human":
                path = f"../data/data_human/{data}_human.csv"
            else:
                path = f"../data/data_ai/{data.replace('-', '_')}/{data}_{model}.csv"

            subset = df[(df["data"] == data) & (df["model"] == model)]
            df_data = pd.read_csv(path)

            idx = subset["index"].tolist()
            df_subset = df_data.iloc[idx]

            label = 0 if model in cols_c0 else 1

            # save df_subset to csv at save_path
            with open(save_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for i in range(len(df_subset)):
                    text = df_subset.iloc[i]["text"]
                    writer.writerow([text, label])
