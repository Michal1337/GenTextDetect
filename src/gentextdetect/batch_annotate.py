"""Batch-annotate every PDF under INPUT_DIR (recursive) and write the
results to OUTPUT_DIR, mirroring the input subfolder structure.

Edit the CONFIG block below and run::

    python src/gentextdetect/batch_annotate.py

Uses the same model + pipeline as the Streamlit PDF tab
(extract → predict_long_text → annotate).
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Model — keep in sync with app.py if you want identical scoring.
MODEL_TYPE: str = "finetune"  # "baseline" or "finetune"
DEVICE: str = "cuda"          # "cpu" or "cuda"
CHECKPOINT_PATH: str = (
    "./checkpoints/finetune/finetuned_model_phi-4_master-large.pt"
)

# Baseline-only
BASELINE_MODEL_SIZE: str = "mini"
BASELINE_TOKENIZER: str = "meta-llama/Llama-3.2-1B-Instruct"

# Finetune-only
MODEL_ROOT: str = "/mnt/evafs/groups/re-com/mgromadzki/llms/"
OVERRIDE_BASE_PATH: str = ""
OVERRIDE_BASE_NAME: str = ""

# IO
INPUT_DIR: str = "./paper_data"
OUTPUT_DIR: str = "./paper_data_annotated"

# If False, files whose annotated output already exists are skipped.
OVERWRITE: bool = False

# Appended to each PDF's stem before .pdf — set to "" to keep original names.
ANNOTATED_SUFFIX: str = "_annotated"

# Drop math-font spans and lines that look like math fragments / plot labels
# before scoring. Reduces equation/figure noise at the cost of occasionally
# dropping legitimate short lines.
CLEAN_TEXT: bool = True

# Number of worker threads. Inference still serializes on the GPU, but the
# CPU-bound annotation + IO of one PDF runs concurrently with the inference
# of the next, so 2-4 workers usually hides annotation cost entirely.
NUM_WORKERS: int = 4

# CSV with one summary row per PDF (path is relative to OUTPUT_DIR if not
# absolute). Set to "" to disable.
SUMMARY_CSV: str = "summary.csv"

# Probability cutoffs counted as separate columns in the summary CSV.
HIGH_PROB_THRESHOLDS: Tuple[float, ...] = (0.5, 0.8, 0.95)

# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "experiments"
)
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _vendor_folder(base_model_name: str) -> Optional[str]:
    if "Meta-Llama-3.1-70B-Instruct-AWQ-INT4" in base_model_name:
        return "hugging-quants"
    if "Falcon" in base_model_name:
        return "tiiuae"
    if "Llama" in base_model_name:
        return "meta-llama"
    if "phi" in base_model_name.lower():
        return "microsoft"
    if "Qwen" in base_model_name:
        return "Qwen"
    if "Ministral" in base_model_name or "Mistral" in base_model_name:
        return "mistralai"
    return None


def _parse_base_model_from_checkpoint(path: str) -> Optional[str]:
    if not path:
        return None
    stem = Path(path).stem
    parts = stem.split("_")
    if (
        len(parts) >= 3
        and parts[0].startswith("finetune")
        and parts[1] == "model"
    ):
        return parts[2]
    if len(parts) >= 2 and parts[0].startswith("finetune"):
        return parts[1]
    return None


def _resolve_finetune_paths() -> Tuple[str, str, bool]:
    base_name = OVERRIDE_BASE_NAME or _parse_base_model_from_checkpoint(
        CHECKPOINT_PATH
    )
    if not base_name:
        raise ValueError(
            "Could not detect base model name from CHECKPOINT_PATH. Set "
            "OVERRIDE_BASE_NAME explicitly."
        )
    if OVERRIDE_BASE_PATH:
        base_path = OVERRIDE_BASE_PATH
    else:
        if not MODEL_ROOT:
            raise ValueError("Set MODEL_ROOT or OVERRIDE_BASE_PATH.")
        vendor = _vendor_folder(base_name)
        if not vendor:
            raise ValueError(
                f"Unknown vendor folder for base model '{base_name}'. "
                "Set OVERRIDE_BASE_PATH."
            )
        base_path = os.path.join(MODEL_ROOT, vendor, base_name)
    is_phi = "phi" in base_name.lower()
    return base_path, base_name, is_phi


def prob_to_rgb(prob: float) -> Tuple[float, float, float]:
    prob = max(0.0, min(1.0, prob))
    if prob < 0.5:
        t = prob / 0.5
        r = (80 + (255 - 80) * t) / 255.0
        g = 200 / 255.0
    else:
        t = (prob - 0.5) / 0.5
        r = 1.0
        g = (200 - (200 - 60) * t) / 255.0
    b = 60 / 255.0
    return (r, g, b)


def load_model_and_tokenizer():
    if MODEL_TYPE == "baseline":
        from inference import load_baseline
        model, tokenizer = load_baseline(
            CHECKPOINT_PATH, BASELINE_MODEL_SIZE, BASELINE_TOKENIZER, DEVICE
        )
        meta = {
            "type": "baseline",
            "size": BASELINE_MODEL_SIZE,
            "checkpoint": CHECKPOINT_PATH,
        }
    elif MODEL_TYPE == "finetune":
        from inference import load_finetune
        base_path, base_name, is_phi = _resolve_finetune_paths()
        model, tokenizer = load_finetune(
            CHECKPOINT_PATH,
            base_path,
            base_name,
            is_phi=is_phi,
            device=DEVICE,
        )
        meta = {
            "type": "finetune (phi)" if is_phi else "finetune",
            "base_model": base_name,
            "base_path": base_path,
            "checkpoint": CHECKPOINT_PATH,
        }
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE!r}")
    return model, tokenizer, meta


def _stats(probs: List[float]) -> Dict[str, Any]:
    if not probs:
        return {
            "n_tokens": 0,
            "total_tokens": 0,
            "mean_prob": None,
            "std_prob": None,
            "min_prob": None,
            "max_prob": None,
            "p25_prob": None,
            "median_prob": None,
            "p75_prob": None,
            "verdict": None,
            **{f"frac_above_{thr}": None for thr in HIGH_PROB_THRESHOLDS},
        }
    arr = np.asarray(probs, dtype=np.float64)
    mean = float(arr.mean())
    out: Dict[str, Any] = {
        "n_tokens": int(arr.size),
        "mean_prob": mean,
        "std_prob": float(arr.std()),
        "min_prob": float(arr.min()),
        "max_prob": float(arr.max()),
        "p25_prob": float(np.percentile(arr, 25)),
        "median_prob": float(np.percentile(arr, 50)),
        "p75_prob": float(np.percentile(arr, 75)),
        "verdict": "ai" if mean >= 0.5 else "human",
    }
    for thr in HIGH_PROB_THRESHOLDS:
        out[f"frac_above_{thr}"] = float((arr >= thr).mean())
    return out


def process_pdf(pdf_path: Path, out_path: Path, model, tokenizer) -> dict:
    from inference import predict_long_text
    from pdf_utils import annotate_pdf, extract_chars

    pdf_bytes = pdf_path.read_bytes()
    text, char_spans = extract_chars(pdf_bytes, clean=CLEAN_TEXT)
    if not text.strip():
        return {"status": "empty", **_stats([])}

    token_probs = predict_long_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    valid = [t.prob for t in token_probs if t.char_end > t.char_start]
    stats = _stats(valid)
    stats["total_tokens"] = len(token_probs)

    annotated = annotate_pdf(
        pdf_bytes=pdf_bytes,
        char_spans=char_spans,
        token_spans=[
            (t.char_start, t.char_end, t.prob) for t in token_probs
        ],
        color_fn=prob_to_rgb,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(annotated)
    return {"status": "ok", **stats}


def _csv_fieldnames() -> List[str]:
    base = [
        "path",
        "status",
        "n_tokens",
        "total_tokens",
        "mean_prob",
        "std_prob",
        "min_prob",
        "max_prob",
        "p25_prob",
        "median_prob",
        "p75_prob",
    ]
    base += [f"frac_above_{thr}" for thr in HIGH_PROB_THRESHOLDS]
    base += ["verdict", "elapsed_s", "error"]
    return base


def _open_summary_csv(out_dir: Path):
    if not SUMMARY_CSV:
        return None, None
    csv_path = Path(SUMMARY_CSV)
    if not csv_path.is_absolute():
        csv_path = out_dir / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fh = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_csv_fieldnames())
    writer.writeheader()
    fh.flush()
    return fh, writer


def main() -> None:
    in_dir = Path(INPUT_DIR).resolve()
    out_dir = Path(OUTPUT_DIR).resolve()

    if not in_dir.is_dir():
        sys.exit(f"INPUT_DIR does not exist: {in_dir}")
    if out_dir == in_dir:
        sys.exit("INPUT_DIR and OUTPUT_DIR must differ.")
    if in_dir in out_dir.parents:
        sys.exit("OUTPUT_DIR must not live inside INPUT_DIR.")
    if MODEL_TYPE not in ("baseline", "finetune"):
        sys.exit(f"MODEL_TYPE must be 'baseline' or 'finetune': {MODEL_TYPE!r}")
    if DEVICE not in ("cpu", "cuda"):
        sys.exit(f"DEVICE must be 'cpu' or 'cuda': {DEVICE!r}")
    if not os.path.isfile(CHECKPOINT_PATH):
        sys.exit(f"CHECKPOINT_PATH does not exist: {CHECKPOINT_PATH}")

    pdfs = sorted(in_dir.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found under {in_dir}")

    print(f"Found {len(pdfs)} PDF(s) under {in_dir}")
    print(f"Writing to {out_dir} (overwrite={OVERWRITE})")
    print(f"Loading model — MODEL_TYPE={MODEL_TYPE}, DEVICE={DEVICE}…")

    try:
        model, tokenizer, meta = load_model_and_tokenizer()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed to load model: {type(exc).__name__}: {exc}")
    print(f"Model ready: {meta}")
    print()

    counts = {"ok": 0, "skipped_existing": 0, "empty": 0, "error": 0}
    csv_fh, csv_writer = _open_summary_csv(out_dir)
    if csv_fh is not None:
        print(f"Summary CSV: {csv_fh.name}")
    print(f"Workers: {NUM_WORKERS}")
    print()

    csv_lock = threading.Lock()
    counts_lock = threading.Lock()
    print_lock = threading.Lock()

    def _write_row(rel_path: Path, status: str, elapsed: Optional[float],
                    payload: Optional[Dict[str, Any]] = None,
                    error: str = "") -> None:
        if csv_writer is None:
            return
        row: Dict[str, Any] = {f: "" for f in _csv_fieldnames()}
        row["path"] = str(rel_path).replace("\\", "/")
        row["status"] = status
        row["error"] = error
        if elapsed is not None:
            row["elapsed_s"] = round(elapsed, 3)
        if payload:
            for k, v in payload.items():
                if k in row and v is not None:
                    if isinstance(v, float):
                        row[k] = round(v, 6)
                    else:
                        row[k] = v
        with csv_lock:
            csv_writer.writerow(row)
            csv_fh.flush()

    width = len(str(len(pdfs)))

    def _process_one(idx: int, pdf_path: Path) -> None:
        rel = pdf_path.relative_to(in_dir)
        out_name = rel.stem + ANNOTATED_SUFFIX + rel.suffix
        out_path = out_dir / rel.parent / out_name
        prefix = f"[{idx:>{width}}/{len(pdfs)}]"

        if out_path.is_file() and not OVERWRITE:
            with counts_lock:
                counts["skipped_existing"] += 1
            with print_lock:
                print(f"{prefix} SKIP (exists)   {rel}")
            _write_row(rel, "skipped_existing", None)
            return

        t0 = time.time()
        try:
            result = process_pdf(pdf_path, out_path, model, tokenizer)
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            with counts_lock:
                counts["error"] += 1
            with print_lock:
                print(
                    f"{prefix} ERROR           {rel}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
            _write_row(
                rel, "error", dt,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        dt = time.time() - t0

        if result["status"] == "ok":
            with counts_lock:
                counts["ok"] += 1
            mean_str = (
                f"{result['mean_prob']:.3f}"
                if result["mean_prob"] is not None
                else "n/a"
            )
            with print_lock:
                print(
                    f"{prefix} OK              {rel} — "
                    f"{result['total_tokens']:>6,} tokens · "
                    f"mean P(AI)={mean_str} · {dt:5.1f}s"
                )
            _write_row(rel, "ok", dt, payload=result)
        elif result["status"] == "empty":
            with counts_lock:
                counts["empty"] += 1
            with print_lock:
                print(
                    f"{prefix} EMPTY           {rel} "
                    "(no extractable text)"
                )
            _write_row(rel, "empty", dt, payload=result)

    t_start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, NUM_WORKERS)) as pool:
            futures = [
                pool.submit(_process_one, i, p)
                for i, p in enumerate(pdfs, start=1)
            ]
            for f in as_completed(futures):
                f.result()  # surface any unhandled exception
    finally:
        if csv_fh is not None:
            csv_fh.close()

    total = time.time() - t_start
    print()
    print(f"Done in {total:.1f}s.")
    print(f"  ok:               {counts['ok']}")
    print(f"  skipped existing: {counts['skipped_existing']}")
    print(f"  empty:            {counts['empty']}")
    print(f"  errored:          {counts['error']}")
    if csv_writer is not None:
        print(f"  summary CSV:      {csv_fh.name}")


if __name__ == "__main__":
    main()
