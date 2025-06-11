import csv
from typing import List

import pandas as pd
from tqdm import tqdm

from ex_params import (
    DATA_AI_FEATURES_PATH,
    DATA_HUMAN_FEATURES_PATH,
    DATASETS,
    DATASETS_PATH,
    SELECTED_FEATURES1,
    SELECTED_FEATURES2,
)


def idx2features(
    df: pd.DataFrame,
    features: List[str],
    cols_c0: List[str],
    reverse_labels: bool,
    save_path: str,
) -> None:
    # init csv
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(features + ["label"])

    # iterate through every data and model combination
    for data in tqdm(df["data"].unique()):
        for model in df["model"].unique():
            if model == "human":
                path = DATA_HUMAN_FEATURES_PATH + f"{data}_human_features.csv"
            else:
                path = (
                    DATA_AI_FEATURES_PATH
                    + f"{data.replace('-', '_')}/{data}_{model}_features.csv"
                )

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
                    row = [df_subset.iloc[i][f] for f in features]
                    writer.writerow(row + [label])


if __name__ == "__main__":
    for name, config in DATASETS.items():
        print(f"Creating features dataset for {name}...")
        max_tokens, cols_c0, reverse_labels = (
            config["num_tokens"],
            config["cols_c0"],
            config["reverse_labels"],
        )
        if name == "master-testset":
            test_idx_path = DATASETS_PATH + f"{name}/test_idx.csv"
            df_idx = pd.read_csv(test_idx_path)
            save_path_features = DATASETS_PATH + f"{name}/test_features.csv"
            idx2features(
                df_idx, SELECTED_FEATURES1, cols_c0, reverse_labels, save_path_features
            )

            save_path_features = DATASETS_PATH + f"{name}/test_features2.csv"
            idx2features(
                df_idx, SELECTED_FEATURES2, cols_c0, reverse_labels, save_path_features
            )
        else:
            train_idx_path = DATASETS_PATH + f"{name}/train_idx.csv"
            val_idx_path = DATASETS_PATH + f"{name}/val_idx.csv"

            df_idx = pd.read_csv(train_idx_path)

            save_path_features = DATASETS_PATH + f"{name}/train_features.csv"
            idx2features(
                df_idx, SELECTED_FEATURES1, cols_c0, reverse_labels, save_path_features
            )

            save_path_features = DATASETS_PATH + f"{name}/train_features2.csv"
            idx2features(
                df_idx, SELECTED_FEATURES2, cols_c0, reverse_labels, save_path_features
            )

            df_idx = pd.read_csv(val_idx_path)

            save_path_features = DATASETS_PATH + f"{name}/val_features.csv"
            idx2features(
                df_idx, SELECTED_FEATURES1, cols_c0, reverse_labels, save_path_features
            )

            save_path_features = DATASETS_PATH + f"{name}/val_features2.csv"
            idx2features(
                df_idx, SELECTED_FEATURES2, cols_c0, reverse_labels, save_path_features
            )
