import argparse
import csv
import time

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ex_params import (CHECKPOINTS_PATH, DATASETS_PATH, PAD_TOKENS, SEED, TRAINING_CONFIG,
                       TRAINING_HISTORY_PATH)
from ex_utils import TextDataset, collate_fn_longest, evaluate_2gpus
from models import FineTuneClassifier2GPUs

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    model_name = args.model_path.split("/")[-1]
    ds_train_path = f"{DATASETS_PATH}/{args.dataset_name}/train.csv"
    ds_val_path = f"{DATASETS_PATH}/{args.dataset_name}/val.csv"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if model_name in PAD_TOKENS.keys():
        tokenizer.pad_token = PAD_TOKENS[model_name]
    tokenizer.padding_side = "left"
    config = TRAINING_CONFIG[model_name]

    df_train = pd.read_csv(ds_train_path)
    df_val = pd.read_csv(ds_val_path)
    train_dataset = TextDataset(df_train["text"].tolist(), df_train["label"].tolist())
    val_dataset = TextDataset(df_val["text"].tolist(), df_val["label"].tolist())

    print("=" * 50)
    print(f"Model name: {model_name}, Dataset name: {args.dataset_name}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_longest(batch, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_longest(batch, tokenizer),
    )

    model = FineTuneClassifier2GPUs(
        base_model_path=args.model_path,
        num_labels=1,
        load_device_idx=0,
        infer_device_idx=1,
    )

    total_batch_size = config["total_batch_size"]
    B = args.batch_size
    assert (
        total_batch_size % B == 0
    ), "make sure total_batch_size is divisible by B"
    grad_accum_steps = total_batch_size // B

    max_lr = config["start_lr"]
    min_lr = max_lr * 0.1
    warmup_steps = int(0.1 * len(train_loader) // grad_accum_steps)
    max_steps = int(0.9 * len(train_loader) // grad_accum_steps)

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps

        if it > max_steps:
            return min_lr

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 1.0 - decay_ratio  # Linearly decays from 1 to 0
        return min_lr + coeff * (max_lr - min_lr)


    print(
        f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"Dataset size: {len(train_loader)}")
    print(f"Batch Size: {B} Grad Accumulation Steps: {grad_accum_steps}")
    print(
        f"Max LR: {max_lr} Min LR: {min_lr} Warmup Steps: {warmup_steps} Max Steps: {max_steps}"
    )

    loss_fn = BCEWithLogitsLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=get_lr(0), betas=(0.9, 0.999), fused=True)

    lr = get_lr(0)
    norm = -1
    best_val_acc = -1
    step = 0
    tokens_per_sec_list = []

    history_path = TRAINING_HISTORY_PATH + f"finetune/training_history_finetune_{model_name}_{args.dataset_name}.csv"
    
    with open(history_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
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
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step, batch in enumerate(train_loader, start=1):
            t0 = time.time()
            input_ids = batch["input_ids"].to(model.load_device)
            attention_mask = batch["attention_mask"].to(model.load_device)
            labels = batch["labels"].to(model.infer_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask)

                mask = labels.view(-1) != -100
                labels = labels.view(-1)[mask].float()
                outputs = outputs.view(-1)[mask]
                loss = loss_fn(outputs, labels) / grad_accum_steps

            loss_accum += loss.detach()
            loss.backward()

            t1 = time.time()
            dt = t1 - t0
            tokens_processed = B * input_ids.shape[1]
            tokens_per_sec = tokens_processed / dt
            tokens_per_sec_list.append(tokens_per_sec)
            if len(tokens_per_sec_list) > 100:
                tokens_per_sec_list.pop(0)
            tokens_per_sec_avg = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
            
            print(
                f"step {micro_step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec_avg:.2f}"
            )

            if micro_step % grad_accum_steps == 0 or micro_step == len(train_loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                epoch_loss += loss_accum.item() * grad_accum_steps
                loss_accum = 0.0
                
        avg_loss = epoch_loss / len(train_loader)

        val_metrics = evaluate_2gpus(model, val_loader)

        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        print("Val Metrics:", val_metrics)

        record = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        # Save training history
        with open(history_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:  # fix, tylko classifier
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                model.classifier.state_dict(),
                CHECKPOINTS_PATH
                + f"finetune/finetuned_model_{model_name}_{args.dataset_name}.pt",
            )
            print(f"New best classifier saved (val accuracy: {best_val_acc:.4f})")

