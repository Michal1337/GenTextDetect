import argparse
import csv
import os

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from ex_params import (BASELINE_MODELS, CHECKPOINTS_PATH, DATASETS_PATH, SEED,
                       TRAINING_HISTORY_PATH)
from ex_utils import TextDataset, collate_fn, evaluate
from models import BaselineClassifier

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("model_size", type=str, help="Size of the model")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    ds_train_path = f"{DATASETS_PATH}/{args.dataset_name}/train.csv"
    ds_val_path = f"{DATASETS_PATH}/{args.dataset_name}/val.csv"
    model_config = BASELINE_MODELS[args.model_size]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    df_train = pd.read_csv(ds_train_path)
    df_val = pd.read_csv(ds_val_path)
    train_dataset = TextDataset(df_train["text"].tolist(), df_train["label"].tolist())
    val_dataset = TextDataset(df_val["text"].tolist(), df_val["label"].tolist())

    init_process_group(backend="nccl")

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"World Size: {ddp_world_size}, Local Rank: {ddp_local_rank}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=True,
        seed=SEED,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        sampler=train_sampler,
    )

    if master_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )

    model = BaselineClassifier(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        nhead=model_config["num_heads"],
        max_seq_length=model_config["max_len"],
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
    )

    model.to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    loss_fn = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4)

    best_val_acc = -1

    history_path = (
        TRAINING_HISTORY_PATH
        + f"baseline/training_history_baseline_{args.model_size}_{args.dataset_name}.csv"
    )
    if master_process:
        with open(history_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "train_balanced_accuracy",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "train_auc",
                    "val_loss",
                    "val_accuracy",
                    "val_balanced_accuracy",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "val_auc",
                ],
            )
            writer.writeheader()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        if master_process:
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, disable=not master_process)

        all_logits = []
        all_labels = []
        all_bin_preds = []

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)

            mask = labels.view(-1) != -100
            labels = labels.view(-1)[mask].float()
            outputs = outputs.view(-1)[mask]

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_description(f"Loss: {loss.item():.4f}")

            # Collect predictions during training
            logits = torch.sigmoid(outputs).squeeze().detach().cpu()
            labels_cpu = labels.squeeze().cpu()
            bin_preds = (logits >= 0.5).long()

            all_logits.extend(logits.tolist())
            all_labels.extend(labels_cpu.tolist())
            all_bin_preds.extend(bin_preds.tolist())

        avg_loss = epoch_loss / len(train_loader)

        if master_process:
            train_metrics = {
                "accuracy": accuracy_score(all_labels, all_bin_preds),
                "balanced_accuracy": balanced_accuracy_score(all_labels, all_bin_preds),
                "precision": precision_score(all_labels, all_bin_preds),
                "recall": recall_score(all_labels, all_bin_preds),
                "f1": f1_score(all_labels, all_bin_preds),
                "auc": roc_auc_score(all_labels, all_logits),
            }

            val_metrics = evaluate(model, val_loader, device, "baseline")

            print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
            print("Train Metrics:", train_metrics)
            print("Val Metrics:", val_metrics)

            record = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            # Save training history
            with open(history_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                writer.writerow(record)

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(
                    raw_model.state_dict(),
                    CHECKPOINTS_PATH
                    + f"baseline/baseline_{args.model_size}_{args.dataset_name}.pt",
                )
                print(f"New best model saved (val accuracy: {best_val_acc:.4f})")

    destroy_process_group()
