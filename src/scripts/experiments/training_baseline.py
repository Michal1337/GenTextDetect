import argparse
import csv
import math
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from ex_params import (
    BASELINE_MODELS,
    CHECKPOINTS_PATH,
    DATASETS_PATH,
    MAX_TEXT_LENGTH,
    SEED,
    TRAINING_HISTORY_PATH,
)
from ex_utils import TextDataset, collate_fn, evaluate
from models import BaselineClassifier

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")


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
    tokenizer.padding_side = "left"

    df_train = pd.read_csv(ds_train_path)
    df_val = pd.read_csv(ds_val_path)
    train_dataset = TextDataset(df_train["text"].tolist(), df_train["label"].tolist())
    val_dataset = TextDataset(df_val["text"].tolist(), df_val["label"].tolist())

    init_process_group(backend="nccl")

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    device_type = "cuda"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    if master_process:
        print("=" * 50)
        print(f"Model size: {args.model_size}, Dataset name: {args.dataset_name}")

    print(f"World Size: {ddp_world_size}, Local Rank: {ddp_local_rank}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=True,
        seed=SEED,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=False,
        seed=SEED,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        sampler=val_sampler,
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

    total_batch_size = 2**19  # 2**19, ~0.5M, in number of tokens
    B = args.batch_size
    T = MAX_TEXT_LENGTH
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    max_lr = model_config["start_lr"]
    min_lr = max_lr * 0.1
    warmup_steps = 2 * len(train_loader) // grad_accum_steps
    max_steps = 8 * len(train_loader) // grad_accum_steps

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    if master_process:
        print(
            f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        print(f"Dataset size: {len(train_loader)}")
        print(f"Batch Size: {B} Grad Accumulation Steps: {grad_accum_steps}")
        print(
            f"Max LR: {max_lr} Min LR: {min_lr} Warmup Steps: {warmup_steps} Max Steps: {max_steps}"
        )

    loss_fn = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=get_lr(0), betas=(0.9, 0.95), fused=True)

    lr = get_lr(0)
    norm = -1
    best_val_acc = -1
    step = 0

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

        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step, batch in enumerate(train_loader, start=1):
            t0 = time.time()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(input_ids)

                mask = labels.view(-1) != -100
                labels = labels.view(-1)[mask].float()
                outputs = outputs.view(-1)[mask]
                loss = loss_fn(outputs, labels)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            t1 = time.time()
            dt = t1 - t0
            tokens_processed = B * T * ddp_world_size
            tokens_per_sec = tokens_processed / dt
            if master_process:
                print(
                    f"step {micro_step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                )

            if micro_step % grad_accum_steps == 0 or micro_step == len(train_loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                step += 1
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                epoch_loss += loss_accum.item() * grad_accum_steps
                loss_accum = 0.0

        avg_loss = epoch_loss / len(train_loader)

        torch.cuda.empty_cache()
        val_metrics = evaluate(model, val_loader, device, "baseline", master_process)

        if master_process:
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
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(
                    raw_model.state_dict(),
                    CHECKPOINTS_PATH
                    + f"baseline/baseline_{args.model_size}_{args.dataset_name}.pt",
                )
                print(f"New best model saved (val accuracy: {best_val_acc:.4f})")

    destroy_process_group()
