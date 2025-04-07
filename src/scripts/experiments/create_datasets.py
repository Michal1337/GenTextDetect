import csv
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ex_params import (DATA_AI_PATH, DATA_HUMAN_PATH, DATASETS, DATASETS_PATH,
                       MASTER_STATS_PATH, STATS_PATH)
from ex_utils import get_csv_paths

BATCH_SIZE = 256


def remove_test_samples(
    stats: Dict[str, pd.DataFrame], test_idx: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    for data in tqdm(test_idx["data"].unique()):
        for model in test_idx["model"].unique():
            subset_idx = test_idx[
                (test_idx["data"] == data) & (test_idx["model"] == model)
            ]
            stats[f"{data}_{model}"].drop(subset_idx["index"], inplace=True)

    return stats


def get_master_stats(stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    master_stats = {
        "data": [],
        "model": [],
        "num_samples": [],
        "num_sentences": [],
        "num_words": [],
        "num_chars": [],
        "num_tokens": [],
    }
    for k, v in stats.items():
        data, model = k.split("_")
        master_stats["data"].append(data)
        master_stats["model"].append(model)
        master_stats["num_samples"].append(len(v))
        for col in v.columns:
            master_stats[col].append(v[col].sum())
    df = pd.DataFrame(master_stats)
    return df


def calculate_probs(df_main: pd.DataFrame, cols_c0: List[str]) -> pd.DataFrame:
    df_main["avg_token_per_sample"] = df_main["num_tokens"] / df_main["num_samples"]

    for ds in df_main["data"].unique():
        df_main.loc[df_main["data"].values == ds, "prob"] = (
            df_main.loc[df_main["data"].values == ds, "avg_token_per_sample"].values
            / df_main.loc[df_main["data"].values == ds, "avg_token_per_sample"].sum()
        )
        mask_c0 = (df_main["data"].values == ds) & (df_main["model"].isin(cols_c0))
        mask_c1 = (df_main["data"].values == ds) & (~df_main["model"].isin(cols_c0))

        class0 = df_main[mask_c0]
        class1 = df_main[mask_c1]

        s1 = (class0["avg_token_per_sample"] * class0["prob"]).sum()
        s2 = (class1["avg_token_per_sample"] * class1["prob"]).sum()
        p1 = class0["prob"].sum()
        p2 = class1["prob"].sum()

        c1 = 1 / (s2 / s1 * p1 + p2)
        c0 = c1 * s2 / s1

        df_main.loc[mask_c0, "prob"] *= c0
        df_main.loc[mask_c1, "prob"] *= c1

    return df_main


def create_dataset_idx(
    max_tokens: int,
    batch_size: int,
    stats: Dict[str, pd.DataFrame],
    df_main: pd.DataFrame,
    save_path: str,
) -> None:
    weights = [
        df_main.loc[df_main["data"] == ds, "num_tokens"].sum()
        for ds in df_main["data"].unique()
    ]

    # weights = [
    #     (
    #         df_main.loc[df_main["data"] == ds, "num_tokens"]
    #         * df_main.loc[df_main["data"] == ds, "prob"]
    #     ).sum()
    #     for ds in df_main["data"].unique()
    # ]
    probs = np.array(weights) / np.sum(weights)

    total_tokens = 0
    total_sentences = 0
    total_samples = 0
    cnt = 0
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


def idx2csv(
    df: pd.DataFrame, cols_c0: List[str], reverse_labels: bool, save_path: str
) -> None:
    # init csv
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

    # iterate through every data and model combination
    for data in tqdm(df["data"].unique()):
        for model in df["model"].unique():
            if model == "human":
                path = DATA_HUMAN_PATH + f"{data}_human.csv"
            else:
                path = DATA_AI_PATH + f"{data.replace('-', '_')}/{data}_{model}.csv"

            subset = df[(df["data"] == data) & (df["model"] == model)]
            df_data = pd.read_csv(path)

            idx = subset["index"].tolist()
            df_subset = df_data.iloc[idx]

            if reverse_labels:
                label = 1 if model in cols_c0 else 0
            else:
                label = 0 if model in cols_c0 else 1

            # save df_subset to csv at save_path
            with open(save_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for i in range(len(df_subset)):
                    text = df_subset.iloc[i]["text"]
                    writer.writerow([text, label])

        df_main = pd.read_csv(MASTER_STATS_PATH)
        df_main["avg_token_per_sample"] = df_main["num_tokens"] / df_main["num_samples"]


if __name__ == "__main__":
    paths = get_csv_paths(STATS_PATH, recursive=True)

    for name, config in DATASETS.items():
        max_tokens, cols_c0, reverse_labels = (
            config["num_tokens"],
            config["cols_c0"],
            config["reverse_labels"],
        )

        stats = dict(
            {
                f"{path.split("/")[-1].split("_")[0]}_{path.split("/")[-1].split("_")[1]}": pd.read_csv(
                    path
                )
                for path in paths
            }
        )

        test_set_idx_path = DATASETS_PATH + "master_testset/test_idx.csv"
        test_set_idx = pd.read_csv(test_set_idx_path)

        stats = remove_test_samples(stats, test_set_idx)
        df_main = get_master_stats(stats)
        df_main = calculate_probs(df_main)

        os.mkdir(f"{DATASETS_PATH}{name}/")
        save_path_train_idx = DATASETS_PATH + f"{name}/train_idx.csv"
        save_path_val_idx = DATASETS_PATH + f"{name}/val_idx.csv"
        create_dataset_idx(
            max_tokens, BATCH_SIZE, stats, df_main, save_path_train_idx
        )
        create_dataset_idx(
            int(max_tokens * 0.3),
            BATCH_SIZE,
            stats,
            df_main,
            save_path_val_idx,
        )

        df_idx = pd.read_csv(save_path_train_idx)
        save_path_ds = DATASETS_PATH + f"{name}/train.csv"
        idx2csv(df_idx, cols_c0, reverse_labels, save_path_ds)

        df_idx = pd.read_csv(save_path_val_idx)
        save_path_ds = DATASETS_PATH + f"{name}/val.csv"
        idx2csv(df_idx, cols_c0, reverse_labels, save_path_ds)
