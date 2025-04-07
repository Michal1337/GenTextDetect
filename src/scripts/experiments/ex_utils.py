import os
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        """
        texts: list of texts.
        labels: list of labels for all samples.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
        text = self.texts[idx]
        label = self.labels[idx]

        return {"text": text, "label": label}


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


def collate_fn(
    batch: List[Dict[str, torch.tensor]], tokenizer: AutoTokenizer
) -> Dict[str, torch.tensor]:
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encodings = tokenizer(
        texts, truncation=True, padding="longest", return_tensors="pt"
    )

    labels_padded = [
        torch.where(t == 0, torch.tensor(-100), torch.tensor(label))
        for t, label in zip(encodings["attention_mask"], labels)
    ]
    labels_padded = torch.cat(labels_padded)
    encodings["labels"] = labels_padded

    return encodings
