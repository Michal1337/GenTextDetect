import csv
import os
from typing import List

import pandas as pd
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from ex_params import (
    BASELINE_MODELS,
    CHECKPOINTS_PATH,
    DATASETS_PATH,
    MODEL_PATH,
    PAD_TOKENS,
    PREDICTIONS_PATH,
    SEED,
    TRAINING_HISTORY_PATH,
)
from ex_utils import TextDataset, collate_fn, collate_fn_longest, evaluate_test
from models import BaselineClassifier, FineTuneClassifier, FineTuneClassifierPhi


from torch.utils.data import Sampler

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")


class CustomSampler(Sampler):
    def __init__(self, dataset, rank, num_replicas):
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.total_size = len(self.dataset)
        self.indices = self._get_indices()

    def _get_indices(self):
        samples_per_replica = self.total_size // self.num_replicas
        start_index = self.rank * samples_per_replica
        if self.rank == self.num_replicas - 1:
            end_index = self.total_size
        else:
            end_index = start_index + samples_per_replica
        return list(range(start_index, end_index))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_paths(checkpoint_path: str) -> List[str]:
    all_files = []
    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith(".pt"):
                all_files.append(os.path.join(root, file))

    return all_files


def base_model2folder(name):
    if "Meta-Llama-3.1-70B-Instruct-AWQ-INT4" in name:
        return "hugging-quants"
    if "Falcon" in name:
        return "tiiuae"
    if "Llama" in name:
        return "meta-llama"
    if "phi" in name.lower():
        return "microsoft"
    if "Qwen" in name:
        return "Qwen"
    if "Ministral" in name or "Mistral" in name:
        return "mistralai"


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
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            num_labels=1,
        )
        state_dict = torch.load(path, map_location="cpu")

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)

    elif "finetune" in path:
        base_model = path.split("_")[2]
        folder = base_model2folder(base_model)
        base_model_path = os.path.join(MODEL_PATH, folder, base_model)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
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
    test_dataset = TextDataset(df_test["text"].tolist(), df_test["label"].tolist())
    test_sampler = CustomSampler(
        test_dataset, rank=ddp_rank, num_replicas=ddp_world_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_func(batch, tokenizer),
        sampler=test_sampler,
    )

    test_loaders.append((test_loader, df_test, "master-testset"))

    for level in range(6):
        df_test = pd.read_csv(
            os.path.join(DATASETS_PATH, f"master-testset-hard/test{level}.csv")
        )
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
    checkpoints = [
        (
            "../../../checkpoints/finetune/finetuned_model_Meta-Llama-3.1-70B-Instruct-AWQ-INT4_detect-Meta-Llama-3.1-70B-Instruct-AWQ-INT4.pt",
            2,
        )
    ]

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
        eval_path = TRAINING_HISTORY_PATH + f"per_llm_eval.csv"

        if not os.path.exists(eval_path):
            with open(eval_path, mode="w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model",
                        "train_dataset",
                        "test_dataset",
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

    for checkpoint_config in checkpoints:
        checkpoint, batch_size = checkpoint_config

        if master_process:
            print("=" * 50)
            print(f"Checkpoint path: {checkpoint}")

        model_type = "baseline" if "baseline" in checkpoint else "finetune"
        model, tokenizer = path2model(checkpoint)
        collate_func = collate_fn if model_type == "baseline" else collate_fn_longest

        test_loaders = get_test_loaders(batch_size, collate_func, tokenizer)
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
                save_path = (
                    save_path.replace(".pt", "")
                    .replace("baseline_", "")
                    .replace("finetuned_model_", "")
                )

                df_preds = pd.DataFrame(preds, columns=["preds"])
                df_preds["data"] = df_test["data"]
                df_preds["model"] = df_test["model"]
                df_preds["label"] = df_test["label"]

                df_preds.to_csv(
                    os.path.join(
                        PREDICTIONS_PATH,
                        f"{model_type}/preds_{test_name}_{save_path}.csv",
                    ),
                    index=False,
                )

                model_name = checkpoint.split("/")[-1].split("_")[-2]
                train_dataset = (
                    checkpoint.split("/")[-1].split("_")[-1].replace(".pt", "")
                )

                record = {
                    "model_name": model_name,
                    "train_dataset": train_dataset,
                    "test_dataset": test_name,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }

                with open(eval_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=record.keys())
                    writer.writerow(record)

    destroy_process_group()
