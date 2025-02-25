import os
from typing import Dict, List

import numpy as np
import pandas as pd

from params import DATA_AI_PATH, DATA_HUMAN_PATH, DATASET_PARAMS
from utils import get_csv_paths


def create_ds_idx(
    stats: Dict[str, pd.DataFrame],
    datas: List[str],
    llms: List[str],
    max_tokens: int,
    batch_size: int,
    save_path: str,
) -> None:
    total_tokens = 0
    total_sentences = 0
    total_samples = 0
    while total_tokens < max_tokens:
        data = np.random.choice(datas)
        p = np.random.rand()
        # if p < 0.5:
        #     model = "human"
        # else:
        #     model = np.random.choice(llms)
        model = "human"
        stat = stats[f"{data}_{model}"]

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

        cnt += 1
        if cnt % 100 == 0:
            print(
                f"total_tokens: {total_tokens}, total_sentences: {total_sentences}, total_samples: {total_samples}"
            )


if __name__ == "__main__":
    paths = get_csv_paths(DATA_HUMAN_PATH) + get_csv_paths(DATA_AI_PATH, recursive=True)
    stats = dict(
        {
            f"{path.split("/")[-1].split("_")[0]}_{path.split("/")[-1].split("_")[1]}": pd.read_csv(
                path
            )
            for path in paths
        }
    )
