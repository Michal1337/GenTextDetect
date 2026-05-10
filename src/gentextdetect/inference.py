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


@dataclass
class TokenSpanProb:
    """Per-token result for long-text inference: keeps raw character offsets."""
    char_start: int
    char_end: int
    prob: float


def predict_long_text(
    text: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    device: str = "cpu",
    chunk_tokens: int = MAX_TEXT_LENGTH,
    progress_cb=None,
) -> List[TokenSpanProb]:
    """Tokenize the full text and run inference over consecutive
    ``chunk_tokens``-sized windows. Returns per-token (char_start, char_end,
    p_ai) for the entire input — long PDFs included.

    BOS is added per chunk so each window matches the model's training
    distribution; chunks past the first will not see preceding context, which
    is an accepted trade-off for inputs longer than the model's max length.
    """
    if not text:
        return []
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "predict_long_text requires a fast tokenizer (offset_mapping)."
        )

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    all_ids: List[int] = enc["input_ids"]
    all_offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    if not all_ids:
        return []

    out: List[TokenSpanProb] = []
    n = len(all_ids)
    for start in range(0, n, chunk_tokens):
        chunk_ids = all_ids[start : start + chunk_tokens]
        chunk_offsets = all_offsets[start : start + chunk_tokens]

        input_ids = torch.tensor(
            [chunk_ids], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )

        with torch.no_grad(), autocast_ctx:
            if model_type == "baseline":
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask)

        probs = torch.sigmoid(logits.float()).squeeze(-1).squeeze(0).cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([float(probs)])

        for (s, e), p in zip(chunk_offsets, probs):
            out.append(TokenSpanProb(int(s), int(e), float(p)))

        if progress_cb is not None:
            progress_cb(min(start + chunk_tokens, n), n)

    return out


def predict_long_text_batched(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    device: str = "cpu",
    batch_size: int = 4,
    chunk_tokens: int = MAX_TEXT_LENGTH,
    progress_cb=None,
) -> List[List[TokenSpanProb]]:
    """Batched variant of :func:`predict_long_text`.

    Tokenizes every ``texts[i]``, splits each into ``chunk_tokens``-sized
    windows, then groups chunks across all texts into batches of
    ``batch_size``. Each batch is left-padded to the longest chunk in it and
    run as a single forward pass. Returns a parallel list to ``texts`` —
    one ``[TokenSpanProb, ...]`` per input.

    The progress callback (if provided) is invoked as
    ``progress_cb(batches_done, total_batches)`` after each batch completes.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "predict_long_text_batched requires a fast tokenizer."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    # Stage A — tokenize + chunk every text.
    flat_chunks: List[Tuple[int, int, List[int], List[Tuple[int, int]]]] = []
    chunk_counts: List[int] = []
    for ti, text in enumerate(texts):
        if not text:
            chunk_counts.append(0)
            continue
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        if not ids:
            chunk_counts.append(0)
            continue
        cnt = 0
        for start in range(0, len(ids), chunk_tokens):
            flat_chunks.append((
                ti,
                cnt,
                ids[start : start + chunk_tokens],
                offs[start : start + chunk_tokens],
            ))
            cnt += 1
        chunk_counts.append(cnt)

    # Per-text per-chunk storage so we can rebuild ordered output at the end.
    per_text_chunk_probs: List[List[Optional[List[float]]]] = [
        [None] * c for c in chunk_counts
    ]
    per_text_chunk_offsets: List[List[Optional[List[Tuple[int, int]]]]] = [
        [None] * c for c in chunk_counts
    ]
    for ti, ci, _, offs in flat_chunks:
        per_text_chunk_offsets[ti][ci] = offs

    n_batches = (len(flat_chunks) + batch_size - 1) // batch_size
    if progress_cb is not None:
        progress_cb(0, n_batches)

    # Stage B — batched inference with left padding.
    for b_i, b_start in enumerate(range(0, len(flat_chunks), batch_size)):
        batch = flat_chunks[b_start : b_start + batch_size]
        max_len = max(len(item[2]) for item in batch)

        padded_ids: List[List[int]] = []
        masks: List[List[int]] = []
        real_lens: List[int] = []
        for _, _, ids, _ in batch:
            r = len(ids)
            pad_len = max_len - r
            padded_ids.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * r)
            real_lens.append(r)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            if model_type == "baseline":
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask)

        probs = torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy()
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        for k, ((ti, ci, _, _), real_len) in enumerate(zip(batch, real_lens)):
            real_probs = probs[k, max_len - real_len : max_len]
            per_text_chunk_probs[ti][ci] = real_probs.tolist()

        if progress_cb is not None:
            progress_cb(b_i + 1, n_batches)

    # Stage C — assemble per-text output preserving chunk order.
    outputs: List[List[TokenSpanProb]] = []
    for ti in range(len(texts)):
        spans: List[TokenSpanProb] = []
        for ci in range(chunk_counts[ti]):
            chunk_probs = per_text_chunk_probs[ti][ci]
            chunk_offs = per_text_chunk_offsets[ti][ci]
            if chunk_probs is None or chunk_offs is None:
                continue
            for (s, e), p in zip(chunk_offs, chunk_probs):
                spans.append(TokenSpanProb(int(s), int(e), float(p)))
        outputs.append(spans)
    return outputs
