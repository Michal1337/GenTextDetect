import os
import pandas as pd

from ex_params import STATS_PATH, MASTER_STATS_PATH, DATASETS_PATH, DATASETS
from ex_utils import get_csv_paths, create_dataset_idx, idx2csv

BATCH_SIZE = 256

if __name__ == "__main__":
    paths = get_csv_paths(STATS_PATH, recursive=True)

    for name, config in DATASETS.items():
        max_tokens, cols_c0 = config

        stats = dict(
            {
                f"{path.split("/")[-1].split("_")[0]}_{path.split("/")[-1].split("_")[1]}": pd.read_csv(
                    path
                )
                for path in paths
            }
        )

        df_main = pd.read_csv(MASTER_STATS_PATH)
        df_main["avg_sent_per_sample"] = (
            df_main["num_sentences"] / df_main["num_samples"]
        )

        os.mkdir(f"{DATASETS_PATH}/{name}/", exist_ok=True)
        save_path_idx = f"{DATASETS_PATH}/{name}/idx.csv" 
        create_dataset_idx(max_tokens, BATCH_SIZE, stats, df_main, cols_c0, save_path_idx)

        df_idx = pd.read_csv(save_path)
        save_path_ds = f"{DATASETS_PATH}/{name}/dataset.csv" 
        idx2csv(df_idx, cols_c0, save_path)

