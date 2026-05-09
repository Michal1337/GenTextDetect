"""Model loading and per-token inference for the GenTextDetect Streamlit app.

The classifiers used in the thesis emit one logit per input token (shape
``(B, T, 1)``). Applying sigmoid yields the per-token probability that the
token belongs to AI-generated text (label=1 in the training pipeline).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

EXPERIMENTS_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "experiments"
)
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from ex_params import BASELINE_MODELS, MAX_TEXT_LENGTH, PAD_TOKENS  # noqa: E402
from models import BaselineClassifier  # noqa: E402


@dataclass
class TokenPrediction:
    text: str
    prob: float
    is_special: bool


def _strip_compile_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod.") :]] = v
        else:
            out[k] = v
    return out


def load_baseline(
    checkpoint_path: str,
    model_size: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str = "cpu",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    if model_size not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline size {model_size!r}. "
            f"Choose one of {list(BASELINE_MODELS)}."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "left"

    cfg = BASELINE_MODELS[model_size]
    model = BaselineClassifier(
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        nhead=cfg["num_heads"],
        max_seq_length=cfg["max_len"],
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(_strip_compile_prefix(state))
    model.eval()
    model.to(device)
    return model, tokenizer


def load_finetune(
    checkpoint_path: str,
    base_model_path: str,
    base_model_name: Optional[str] = None,
    is_phi: bool = False,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load a finetune classifier head on top of a base LLM.

    Requires CUDA + flash-attn since the original model classes hardcode
    ``attn_implementation="flash_attention_2"`` and ``torch.bfloat16``.
    """
    from models import FineTuneClassifier, FineTuneClassifierPhi  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Finetune classifiers depend on flash_attention_2 + bf16 and "
            "must run on a CUDA device."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )
    if base_model_name and base_model_name in PAD_TOKENS:
        tokenizer.pad_token = PAD_TOKENS[base_model_name]
    tokenizer.padding_side = "left"

    cls = FineTuneClassifierPhi if is_phi else FineTuneClassifier
    model = cls.from_classifier_head(
        base_model_path=base_model_path,
        path=checkpoint_path,
        num_labels=1,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def _per_token_spans(
    text: str,
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    offsets: Optional[List[Tuple[int, int]]],
) -> List[Tuple[str, bool]]:
    """Return (display_text, is_special) for each token id."""
    if offsets is not None:
        spans = []
        for tok_id, (start, end) in zip(input_ids, offsets):
            if start == end:
                # Special tokens have an empty offset span.
                spans.append((tokenizer.decode([tok_id]), True))
            else:
                spans.append((text[start:end], False))
        return spans

    # Fallback for slow tokenizers — decode each id in isolation.
    special_ids = set(tokenizer.all_special_ids or [])
    return [
        (tokenizer.decode([tok_id]), tok_id in special_ids)
        for tok_id in input_ids
    ]


def predict_token_probs(
    text: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    device: str = "cpu",
    max_length: int = MAX_TEXT_LENGTH,
) -> List[TokenPrediction]:
    if not text:
        return []

    enc_kwargs = dict(
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if getattr(tokenizer, "is_fast", False):
        enc_kwargs["return_offsets_mapping"] = True

    encoded = tokenizer(text, **enc_kwargs)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    offsets = None
    if "offset_mapping" in encoded:
        offsets = encoded["offset_mapping"][0].tolist()

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            if model_type == "baseline":
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask)

    probs = torch.sigmoid(logits.float()).squeeze(-1).squeeze(0).cpu().numpy()
    if probs.ndim == 0:
        probs = np.array([float(probs)])

    ids = input_ids[0].cpu().tolist()
    spans = _per_token_spans(text, tokenizer, ids, offsets)

    out: List[TokenPrediction] = []
    for (display, is_special), prob in zip(spans, probs):
        out.append(
            TokenPrediction(
                text=display,
                prob=float(prob),
                is_special=is_special,
            )
        )
    return out


def aggregate_probability(predictions: List[TokenPrediction]) -> float:
    """Mean over non-special tokens, matching ``evaluate_test`` semantics."""
    valid = [p.prob for p in predictions if not p.is_special]
    if not valid:
        return 0.0
    return float(np.mean(valid))
