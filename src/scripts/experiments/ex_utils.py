import os
from typing import Dict, List, Union

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ex_params import MAX_TEXT_LENGTH


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


def collate_fn(
    batch: List[Dict[str, torch.tensor]], tokenizer: AutoTokenizer
) -> Dict[str, torch.tensor]:
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=MAX_TEXT_LENGTH,
    )

    labels_padded = [
        torch.where(t == 0, torch.tensor(-100), torch.tensor(label))
        for t, label in zip(encodings["attention_mask"], labels)
    ]
    labels_padded = torch.stack(labels_padded)
    encodings["labels"] = labels_padded

    return encodings


def collate_fn_longest(
    batch: List[Dict[str, torch.tensor]], tokenizer: AutoTokenizer
) -> Dict[str, torch.tensor]:
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        max_length=MAX_TEXT_LENGTH,
    )

    labels_padded = [
        torch.where(t == 0, torch.tensor(-100), torch.tensor(label))
        for t, label in zip(encodings["attention_mask"], labels)
    ]
    labels_padded = torch.stack(labels_padded)
    encodings["labels"] = labels_padded

    return encodings


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    t: str,
    master_process: bool,
) -> Dict[str, float]:
    model.eval()
    loss_fn = BCEWithLogitsLoss()

    preds_local, targets_local = [], []
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not master_process):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if t == "baseline":
                    outputs = model(input_ids)
                elif t == "finetune":
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    raise ValueError(
                        "Invalid training type. Use 'baseline' or 'finetune'."
                    )

                mask = labels.view(-1) != -100
                labels = labels.view(-1)[mask].float()
                outputs = outputs.view(-1)[mask]

                loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            logits = torch.sigmoid(outputs).float().cpu().view(-1).numpy()
            labels = labels.cpu().view(-1).numpy()

            preds_local.extend(logits.tolist())
            targets_local.extend(labels.tolist())

    # Gather predictions and labels from all processes
    world_size = dist.get_world_size()

    preds_tensor = torch.tensor(preds_local, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets_local, dtype=torch.float32, device=device)

    local_size = torch.tensor([preds_tensor.size(0)], device=device)
    sizes_list = [
        torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(sizes_list, local_size)

    # Prepare tensor_list with appropriate sizes
    preds_list = [
        torch.zeros(size, dtype=preds_tensor.dtype, device=device)
        for size in sizes_list
    ]
    targets_list = [
        torch.zeros(size, dtype=targets_tensor.dtype, device=device)
        for size in sizes_list
    ]

    dist.all_gather(preds_list, preds_tensor)
    dist.all_gather(targets_list, targets_tensor)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

    if master_process:
        preds_all = torch.cat(preds_list).cpu().numpy()
        targets_all = torch.cat(targets_list).cpu().numpy().astype(int)

        # targets_all = np.round(np.clip(targets_all, 0, 1)).astype(int)
        bin_preds = (preds_all >= 0.5).astype(int)

        metrics = {
            "loss": total_loss.item() / max(num_batches.item(), 1),
            "accuracy": accuracy_score(targets_all, bin_preds),
            "balanced_accuracy": balanced_accuracy_score(targets_all, bin_preds),
            "precision": precision_score(targets_all, bin_preds),
            "recall": recall_score(targets_all, bin_preds),
            "f1": f1_score(targets_all, bin_preds),
            "auc": roc_auc_score(targets_all, preds_all),
        }

        return metrics

    return {}  # Other ranks return empty


def evaluate_2gpus(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> Dict[str, float]:
    model.eval()
    loss_fn = BCEWithLogitsLoss()

    preds_local, targets_local = [], []
    total_loss = torch.tensor(0.0)
    num_batches = torch.tensor(0.0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.load_device)
            attention_mask = batch["attention_mask"].to(model.load_device)
            labels = batch["labels"].to(model.infer_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask)

                mask = (labels.view(-1) != -100).cpu()
                labels = labels.view(-1)[mask].float()
                outputs = outputs.view(-1)[mask].to(model.infer_device)

                loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            logits = torch.sigmoid(outputs).squeeze().float().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            preds_local.extend(logits.tolist())
            targets_local.extend(labels.tolist())

    # Gather predictions and labels from all processes
    preds_all = torch.tensor(preds_local, dtype=torch.float32).numpy()
    targets_all = torch.tensor(targets_local, dtype=torch.float32).numpy()

    targets_all = np.round(np.clip(targets_all, 0, 1)).astype(int)
    bin_preds = (preds_all >= 0.5).astype(int)

    metrics = {
        "loss": total_loss.item() / max(num_batches.item(), 1),
        "accuracy": accuracy_score(targets_all, bin_preds),
        "balanced_accuracy": balanced_accuracy_score(targets_all, bin_preds),
        "precision": precision_score(targets_all, bin_preds),
        "recall": recall_score(targets_all, bin_preds),
        "f1": f1_score(targets_all, bin_preds),
        "auc": roc_auc_score(targets_all, preds_all),
    }

    return metrics


def evaluate_test(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    t: str,
    master_process: bool,
) -> Dict[str, float]:
    model.eval()
    loss_fn = BCEWithLogitsLoss()

    preds_local, targets_local, preds_sample_local = [], [], []
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not master_process):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if t == "baseline":
                    outputs = model(input_ids)
                elif t == "finetune":
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    raise ValueError(
                        "Invalid training type. Use 'baseline' or 'finetune'."
                    )

                mask = labels.view(-1) != -100
                labels_flat = labels.view(-1)[mask].float()
                outputs_flat = outputs.view(-1)[mask]

                loss = loss_fn(outputs_flat, labels_flat)

            mask_outputs = (labels != -100).cpu()
            outputs = torch.sigmoid(outputs.float()).squeeze(-1).cpu()
            masked_outputs = torch.where(mask_outputs, outputs, torch.tensor(0.0))
            row_sums = masked_outputs.sum(dim=1)
            valid_counts = mask_outputs.sum(dim=1)
            mean_per_row = (row_sums / valid_counts).cpu().numpy()

            total_loss += loss.item()
            num_batches += 1

            logits = torch.sigmoid(outputs_flat).float().cpu().view(-1).numpy()
            labels_flat = labels_flat.cpu().view(-1).numpy()

            preds_local.extend(logits.tolist())
            targets_local.extend(labels_flat.tolist())
            preds_sample_local.extend(mean_per_row.tolist())

    # Gather predictions and labels from all processes
    preds_tensor = torch.tensor(preds_local, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets_local, dtype=torch.float32, device=device)
    preds_sample_tensor = torch.tensor(
        preds_sample_local, dtype=torch.float32, device=device
    )

    world_size = dist.get_world_size()

    local_pred_size = torch.tensor([preds_tensor.size(0)], device=device)
    local_sample_size = torch.tensor([preds_sample_tensor.size(0)], device=device)

    pred_sizes = [
        torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    sample_sizes = [
        torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
    ]

    dist.all_gather(pred_sizes, local_pred_size)
    dist.all_gather(sample_sizes, local_sample_size)

    # Prepare tensor_list with appropriate sizes
    preds_list = [
        torch.zeros(size, dtype=preds_tensor.dtype, device=device)
        for size in pred_sizes
    ]
    targets_list = [
        torch.zeros(size, dtype=targets_tensor.dtype, device=device)
        for size in pred_sizes
    ]
    preds_sample_list = [
        torch.zeros(size, dtype=preds_sample_tensor.dtype, device=device)
        for size in sample_sizes
    ]

    dist.all_gather(preds_list, preds_tensor)
    dist.all_gather(targets_list, targets_tensor)
    dist.all_gather(preds_sample_list, preds_sample_tensor)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

    if master_process:
        preds_all = torch.cat(preds_list).cpu().numpy()
        targets_all = torch.cat(targets_list).cpu().numpy().astype(int)
        preds_sample_all = torch.cat(preds_sample_list).cpu().numpy()

        # targets_all = np.round(np.clip(targets_all, 0, 1)).astype(int)
        bin_preds = (preds_all >= 0.5).astype(int)

        metrics = {
            "loss": total_loss.item() / max(num_batches.item(), 1),
            "accuracy": accuracy_score(targets_all, bin_preds),
            "balanced_accuracy": balanced_accuracy_score(targets_all, bin_preds),
            "precision": precision_score(targets_all, bin_preds),
            "recall": recall_score(targets_all, bin_preds),
            "f1": f1_score(targets_all, bin_preds),
            "auc": roc_auc_score(targets_all, preds_all),
        }

        return metrics, preds_sample_all

    return None, None


def evaluate_2gpus_test(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> Dict[str, float]:
    model.eval()
    loss_fn = BCEWithLogitsLoss()

    preds_local, targets_local, preds_sample_local = [], [], []
    total_loss = torch.tensor(0.0)
    num_batches = torch.tensor(0.0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.load_device)
            attention_mask = batch["attention_mask"].to(model.load_device)
            labels = batch["labels"].to(model.infer_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask)

                mask = (labels.view(-1) != -100).cpu()
                labels = labels.view(-1)[mask].float()
                outputs = outputs.view(-1)[mask].to(model.infer_device)

                loss = loss_fn(outputs, labels)

            mask_outputs = labels != -100
            outputs = torch.sigmoid(outputs)
            masked_outputs = torch.where(mask_outputs, outputs, torch.tensor(0.0))
            row_sums = masked_outputs.sum(dim=1)
            valid_counts = mask_outputs.sum(dim=1)
            mean_per_row = (row_sums / valid_counts).cpu().numpy()

            total_loss += loss.item()
            num_batches += 1

            logits = torch.sigmoid(outputs).squeeze().float().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            preds_local.extend(logits.tolist())
            targets_local.extend(labels.tolist())
            preds_sample_local.extend(mean_per_row.tolist())

    # Gather predictions and labels from all processes
    preds_all = torch.tensor(preds_local, dtype=torch.float32).numpy()
    targets_all = torch.tensor(targets_local, dtype=torch.float32).numpy()
    preds_sample_all = torch.tensor(preds_sample_local, dtype=torch.float32).numpy()

    targets_all = np.round(np.clip(targets_all, 0, 1)).astype(int)
    bin_preds = (preds_all >= 0.5).astype(int)

    metrics = {
        "loss": total_loss.item() / max(num_batches.item(), 1),
        "accuracy": accuracy_score(targets_all, bin_preds),
        "balanced_accuracy": balanced_accuracy_score(targets_all, bin_preds),
        "precision": precision_score(targets_all, bin_preds),
        "recall": recall_score(targets_all, bin_preds),
        "f1": f1_score(targets_all, bin_preds),
        "auc": roc_auc_score(targets_all, preds_all),
    }

    return metrics, preds_sample_all
