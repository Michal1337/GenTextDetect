import pandas as pd

from params import STATS_PATH, MASTER_STATS_PATH, DATASET_IDX_PATH, MASTER_DATASET_SIZES
from utils import get_csv_paths, create_dataset_idx

BATCH_SIZE = 256

if __name__ == "__main__":
    paths = get_csv_paths(STATS_PATH, recursive=True)
    col_c0 = "human"

    for name, max_tokens in MASTER_DATASET_SIZES.items():

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

        save_path = DATASET_IDX_PATH + f"main_dataset_{name}_idx.csv"
        create_dataset_idx(max_tokens, BATCH_SIZE, stats, df_main, col_c0, save_path)
