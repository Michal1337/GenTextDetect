import argparse

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ex_params import BASELINE_MODELS, DATASETS_PATH
from ex_utils import TextDataset, collate_fn
from models import BaselineModel


def train(model, dataloader, epochs, save_path, logs_path):
    pass


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("model_size", type=str, help="Size of the model")
    parser.add_argument("dataset_size", type=str, help="Size of the dataset")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("batch_size", type=int, help="Batch size")

    args = parser.parse_args()
    ds_path = f"{DATASETS_PATH}/{args.dataset_size}/dataset.csv"
    model_config = BASELINE_MODELS[args.model_size]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    df_data = pd.read_csv(ds_path)
    dataset = TextDataset(df_data["text"].tolist(), df_data["label"].tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    model = BaselineModel(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        nhead=model_config["num_head"],
        max_seq_length=model_config["max_len"],
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
    )
