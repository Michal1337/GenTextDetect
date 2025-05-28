import os
import csv
from typing import List
from transformers import AutoTokenizer
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

from models import FineTuneClassifier, FineTuneClassifierPhi, BaselineClassifier
from ex_utils import collate_fn, collate_fn_longest, TextDataset, evaluate
from ex_params import CHECKPOINTS_PATH, DATASETS_PATH, BASELINE_MODELS, MODEL_PATH, PREDICTIONS_PATH, TRAINING_HISTORY_PATH, SEED, PAD_TOKENS

import os
from typing import Dict, List, Union

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ex_params import MAX_TEXT_LENGTH



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

    
    pred_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    sample_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]

    dist.all_gather(pred_sizes, local_pred_size)
    dist.all_gather(sample_sizes, local_sample_size)

    # Prepare tensor_list with appropriate sizes
    preds_list = [torch.zeros(size, dtype=preds_tensor.dtype, device=device) for size in pred_sizes]
    targets_list = [torch.zeros(size, dtype=targets_tensor.dtype, device=device) for size in pred_sizes]
    preds_sample_list = [torch.zeros(size, dtype=preds_sample_tensor.dtype, device=device) for size in sample_sizes]

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



def get_paths(checkpoint_path: str) -> List[str]:
    all_files = []
    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith(".pt"):
                all_files.append(os.path.join(root, file))

    return all_files


def path2model(path: str):
    if "baseline" in path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "left"

        model_size = path.split("_")[1]
        model_config = BASELINE_MODELS[model_size]
        model = BaselineClassifier(
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            nhead=model_config["num_heads"],
            max_seq_length=model_config["max_len"],
            vocab_size = len(tokenizer),
            pad_token_id = tokenizer.pad_token_id,
            num_labels = 1,
        )
        state_dict = torch.load(path, map_location="cpu")

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)

    elif "finetune" in path:
        base_model = path.split("_")[2]
        base_model_path = os.path.join(MODEL_PATH, base_model)

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if base_model in PAD_TOKENS.keys():
            tokenizer.pad_token = PAD_TOKENS[base_model]
        tokenizer.padding_side = "left"

        if "phi" in path.lower():
            model = FineTuneClassifierPhi.from_classifier_head(
                base_model_path=base_model_path,
                path=path,
                num_labels=1,
            )
        else:
            model = FineTuneClassifier.from_classifier_head(
                base_model_path=base_model_path,
                path=path,
                num_labels=1,
            )
    else:
        raise ValueError("Unknown model type")

    return model, tokenizer


def get_test_loaders(batch_size, collate_func, tokenizer):
    test_loaders = []
    df_test = pd.read_csv(os.path.join(DATASETS_PATH, "master-testset/test.csv"))
    df_test = df_test.head(100)
    test_dataset = TextDataset(df_test["text"].tolist(), df_test["label"].tolist())
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        shuffle=False,
        seed=SEED,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_func(batch, tokenizer),
        sampler=test_sampler,
    )

    test_loaders.append((test_loader, df_test, "master-testset"))

    for level in range(1):
        df_test = pd.read_csv(os.path.join(DATASETS_PATH, f"master-testset-hard/test{level}.csv"))
        test_dataset = TextDataset(df_test["text"].tolist(), df_test["label"].tolist())
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False,
            seed=SEED,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_func(batch, tokenizer),
            sampler=test_sampler,
        )
        test_loaders.append((test_loader, df_test, f"master-testset-hard-{level}"))

    return test_loaders


if __name__ == "__main__":
    checkpoints = get_paths(CHECKPOINTS_PATH)

    init_process_group(backend="nccl")

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    device_type = "cuda"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"World Size: {ddp_world_size}, Local Rank: {ddp_local_rank}")

    if master_process:
        eval_path = TRAINING_HISTORY_PATH + f"test_eval.csv"
    
        with open(eval_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "train_dataset",
                    "test_dataset"
                    "test_loss",
                    "test_accuracy",
                    "test_balanced_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_auc",
                ],
            )
            writer.writeheader()

    for checkpoint in checkpoints:
        if any(sub in checkpoint for sub in ["Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "Meta-Llama-3.3-70B-Instruct-AWQ-INT4", "Qwen2-72B-Instruct-AWQ", "Qwen2.5-72B-Instruct-AWQ"]):
            continue

        if master_process:
            print("=" * 50)
            print(f"Checkpoint path: {checkpoint}")

        model_type = "baseline" if "baseline" in checkpoint else "finetune"
        model, tokenizer = path2model(checkpoint)
        collate_func = collate_fn if model_type == "baseline" else collate_fn_longest
        
        test_loaders = get_test_loaders(4, collate_func, tokenizer)
        model = model.to(device)
        if model_type == "baseline":
            model = torch.compile(model)
        model = DDP(model, device_ids=[ddp_local_rank])

        for test_loader, df_test, test_name in test_loaders:
            metrics, preds = evaluate_test(
                model,
                test_loader,
                device,
                model_type,
                master_process,
            )

            if master_process:

                save_path = checkpoint.split("/")[-1]
                save_path = save_path.replace(".pt", "").replace("baseline_", "").replace("finetuned_model_", "")

                df_preds = pd.DataFrame(preds, columns=["preds"])
   
                df_preds.to_csv(os.path.join(PREDICTIONS_PATH, f"{model_type}/preds_{test_name}_{save_path}.csv"), index=False)

                model_name = checkpoint.split("/")[-1].split("_")[-2]
                train_dataset = checkpoint.split("/")[-1].split("_")[-1]

                record = {
                    "model_name": model_name,
                    "train_dataset": train_dataset,
                    "test_dataset": test_name,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }

                with open(eval_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=record.keys())
                    writer.writerow(record)