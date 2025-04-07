import os
from typing import Dict, List, Union

import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
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


def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: str, type: str
) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    loss_fn = BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if type == "baseline":
                outputs = model(input_ids)
            elif type == "finetune":
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                raise Exception(
                    "Wrong training type, should be 'baseline' or 'finetune'."
                )

            mask = labels.view(-1) != -100
            labels = labels.view(-1)[mask].float()
            outputs = outputs.view(-1)[mask]

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            logits = torch.sigmoid(outputs).squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            preds.extend(logits)
            targets.extend(labels)

    bin_preds = [1 if p >= 0.5 else 0 for p in preds]

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(targets, bin_preds),
        "balanced_accuracy": balanced_accuracy_score(targets, bin_preds),
        "precision": precision_score(targets, bin_preds),
        "recall": recall_score(targets, bin_preds),
        "f1": f1_score(targets, bin_preds),
        "auc": roc_auc_score(targets, preds),
    }

    return metrics
