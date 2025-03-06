import pandas as pd

from params import STATS_PATH, MASTER_STATS_PATH, DATASET_IDX_PATH, MAIN_DATASET_SIZES
from utils import get_csv_paths, create_dataset_idx

BATCH_SIZE = 256

if __name__ == "__main__":
    paths = get_csv_paths(STATS_PATH, recursive=True)
    stats = dict(
        {
            f"{path.split("/")[-1].split("_")[0]}_{path.split("/")[-1].split("_")[1]}": pd.read_csv(
                path
            )
            for path in paths
        }
    )

    df_main = pd.read_csv(MASTER_STATS_PATH)
    df_main["avg_sent_per_sample"] = df_main["num_sentences"] / df_main["num_samples"]

    df_human = df_main[df_main["model"] == "human"]
    df_ai = df_main[df_main["model"] != "human"]

    df_human["prob"] = (
        1
        / df_human["avg_sent_per_sample"]
        / (1 / df_human["avg_sent_per_sample"]).sum()
    )
    df_ai["prob"] = (
        1 / df_ai["avg_sent_per_sample"] / (1 / df_ai["avg_sent_per_sample"]).sum()
    )

    p_human = (df_human["avg_sent_per_sample"] * df_human["prob"]).sum()
    p_ai = (df_ai["avg_sent_per_sample"] * df_ai["prob"]).sum()

    for name, max_tokens in MAIN_DATASET_SIZES.items():
        save_path = DATASET_IDX_PATH + f"main_dataset_{name}_idx.csv"
        create_dataset_idx(
            max_tokens, BATCH_SIZE, stats, df_human, df_ai, p_human, p_ai, save_path
        )
