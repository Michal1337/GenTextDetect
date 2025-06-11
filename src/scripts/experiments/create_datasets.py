import csv
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ex_params import (
    DATA_AI_PATH,
    DATA_HUMAN_PATH,
    DATASETS,
    DATASETS_PATH,
    MAX_TEXT_LENGTH,
    SEED,
    STATS_PATH,
)
from ex_utils import get_csv_paths

np.random.seed(SEED)
BATCH_SIZE = 16


def remove_long_texts(
    stats: Dict[str, pd.DataFrame], max_len: int
) -> Dict[str, pd.DataFrame]:
    for k, v in stats.items():
        stats[k] = v[v["num_tokens"] <= max_len]
    return stats


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
            1 / df_main.loc[df_main["data"].values == ds, "avg_token_per_sample"].values
        ) / (
            1 / df_main.loc[df_main["data"].values == ds, "avg_token_per_sample"]
        ).sum()
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
        (
            df_main.loc[df_main["data"] == ds, "num_tokens"]
            * df_main.loc[df_main["data"] == ds, "prob"]
        ).sum()
        for ds in df_main["data"].unique()
    ]
    probs = np.array(weights) / np.sum(weights)

    empty_datas = []
    total_tokens, total_sentences, total_samples = 0, 0, 0
    cnt, err_cnt = 0, 0
    while total_tokens < max_tokens:
        data = np.random.choice(df_main["data"].unique(), p=probs)
        tmp = df_main[(df_main["data"] == data)]
        model = np.random.choice(tmp["model"], p=tmp["prob"])

        try:
            stat = stats[f"{data}_{model}"]
            slct = stat.sample(n=batch_size, replace=False, random_state=SEED)
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
        except ValueError:
            try:
                # print(f"Empty data: {data}, model: {model}")
                err_cnt += 1

                empty_datas.append(data)
                non_empty_datas = [
                    d for d in df_main["data"].unique() if d not in empty_datas
                ]
                weights_non_empty = [
                    (
                        df_main.loc[df_main["data"] == ds, "num_tokens"]
                        * df_main.loc[df_main["data"] == ds, "prob"]
                    ).sum()
                    for ds in non_empty_datas
                ]
                probs_non_empty = np.array(weights_non_empty) / np.sum(
                    weights_non_empty
                )
                data_new = np.random.choice(non_empty_datas, p=probs_non_empty)
                mult = (
                    df_main[(df_main["data"] == data) & (df_main["model"] == model)][
                        "avg_token_per_sample"
                    ].tolist()[0]
                    / df_main[
                        (df_main["data"] == data_new) & (df_main["model"] == model)
                    ]["avg_token_per_sample"].tolist()[0]
                )
                stat = stats[f"{data_new}_{model}"]
                slct = stat.sample(
                    n=int(batch_size * mult), replace=False, random_state=SEED
                )
                stat.drop(slct.index, inplace=True)

                total_tokens += slct.sum()["num_tokens"]
                total_sentences += slct.sum()["num_sentences"]
                total_samples += int(batch_size * mult)

                # save data, model, slct.index to csv
                slct["data"] = data_new
                slct["model"] = model
                slct.reset_index(inplace=True)
                slct.to_csv(
                    save_path,
                    mode="a",
                    header=not os.path.exists(save_path),
                    index=False,
                )
            except ValueError:
                empty_datas.append(data)
                err_cnt += 1

        if cnt % 1000 == 0:
            print(
                f"total_tokens: {total_tokens}, total_sentences: {total_sentences}, total_samples: {total_samples}"
            )

    print(
        f"Final samples: {total_samples}, Final sentences: {total_sentences}, Final tokens: {total_tokens}, Unsuccessful samples: {err_cnt}"
    )


def idx2csv(
    df: pd.DataFrame, cols_c0: List[str], reverse_labels: bool, save_path: str
) -> None:
    # init csv
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label", "data", "model"])

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
                    writer.writerow([text, label, data, model])


if __name__ == "__main__":
    paths = get_csv_paths(STATS_PATH + "data_ai/", recursive=True) + get_csv_paths(
        STATS_PATH + "data_human/"
    )

    for name, config in DATASETS.items():
        print(f"Creating dataset {name}...")
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
        stats = remove_long_texts(stats, MAX_TEXT_LENGTH)
        if name == "master-testset":
            df_main = get_master_stats(stats)
            df_main = calculate_probs(df_main, cols_c0)
            os.mkdir(f"{DATASETS_PATH}{name}/")
            test_set_idx_path = DATASETS_PATH + f"{name}/test_idx.csv"
            create_dataset_idx(
                max_tokens, BATCH_SIZE, stats, df_main, test_set_idx_path
            )
            save_path_ds = DATASETS_PATH + f"{name}/test.csv"
            df_idx = pd.read_csv(test_set_idx_path)
            idx2csv(df_idx, cols_c0, False, save_path_ds)

        else:
            test_set_idx_path = DATASETS_PATH + "master-testset/test_idx.csv"
            test_set_idx = pd.read_csv(test_set_idx_path)

            stats = remove_test_samples(stats, test_set_idx)

            df_main = get_master_stats(stats)
            df_main = calculate_probs(df_main, cols_c0)

            os.mkdir(f"{DATASETS_PATH}{name}/")
            save_path_train_idx = DATASETS_PATH + f"{name}/train_idx.csv"
            create_dataset_idx(
                max_tokens, BATCH_SIZE, stats, df_main, save_path_train_idx
            )

            # recalculate probs
            df_main = get_master_stats(stats)
            df_main = calculate_probs(df_main, cols_c0)

            save_path_val_idx = DATASETS_PATH + f"{name}/val_idx.csv"
            create_dataset_idx(
                int(max_tokens * 0.3),
                BATCH_SIZE,
                stats,
                df_main,
                save_path_val_idx,
            )
            # create datatsets from indexes
            df_idx = pd.read_csv(save_path_train_idx)
            save_path_ds = DATASETS_PATH + f"{name}/train.csv"
            idx2csv(df_idx, cols_c0, reverse_labels, save_path_ds)

            df_idx = pd.read_csv(save_path_val_idx)
            save_path_ds = DATASETS_PATH + f"{name}/val.csv"
            idx2csv(df_idx, cols_c0, reverse_labels, save_path_ds)
