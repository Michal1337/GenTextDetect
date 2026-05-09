"""Streamlit UI for token-level AI-generated text detection.

Run with::

    streamlit run src/gentextdetect/app.py

All model configuration lives in the CONFIG block at the top of this file —
edit those constants instead of using a sidebar. Heavy imports (torch,
transformers) are deferred until the first Analyze click so the UI paints
immediately.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these constants to point at your model.
# ─────────────────────────────────────────────────────────────────────────────

# "baseline" or "finetune". Phi vs non-phi finetune is auto-detected from the
# checkpoint filename.
MODEL_TYPE: str = "finetune"

# "cpu" or "cuda". Finetune classifiers require "cuda" + flash-attn.
DEVICE: str = "cuda"

# Path to the trained classifier .pt file.
CHECKPOINT_PATH: str = (
    "./checkpoints/finetuned/finetuned_model_phi-4_detect-phi-4.pt"
)

# ── Baseline-only settings (ignored for finetune) ────────────────────────────
BASELINE_MODEL_SIZE: str = "mini"  # one of BASELINE_MODELS keys
BASELINE_TOKENIZER: str = "meta-llama/Llama-3.2-1B-Instruct"

# ── Finetune-only settings (ignored for baseline) ────────────────────────────
# Root containing <vendor>/<base_model>/ subfolders. The full base model path
# is derived as <MODEL_ROOT>/<vendor>/<auto_detected_base_model>/.
MODEL_ROOT: str = "/mnt/evafs/groups/re-com/mgromadzki/llms/"

# Optional manual overrides — leave empty to auto-detect from the checkpoint
# filename (pattern: ``finetuned_model_<base>_<dataset>.pt``).
OVERRIDE_BASE_PATH: str = ""
OVERRIDE_BASE_NAME: str = ""

# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "experiments"
)
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _vendor_folder(base_model_name: str) -> Optional[str]:
    """Mirror of ex_utils' base_model2folder."""
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
    """Recover the base model name from filenames like
    ``finetuned_model_<base>_<dataset>.pt`` (matches evaluation.py)."""
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


def _resolve_finetune_paths():
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


def prob_to_rgb(prob: float) -> tuple[float, float, float]:
    """Green (low prob = human) → yellow → red (high prob = AI), in [0, 1]."""
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


def prob_to_color(prob: float) -> str:
    r, g, b = prob_to_rgb(prob)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.55)"


def render_tokens_html(predictions) -> str:
    pieces: list[str] = []
    for tok in predictions:
        if tok.is_special:
            continue
        display = tok.text
        if not display:
            continue
        color = prob_to_color(tok.prob)
        safe = (
            html.escape(display)
            .replace("\n", "<br>")
            .replace(" ", "&nbsp;")
        )
        pieces.append(
            f'<span title="p(AI)={tok.prob:.3f}" '
            f'style="background-color:{color}; padding:2px 1px; '
            f'border-radius:3px; line-height:1.9;">{safe}</span>'
        )
    return (
        '<div style="font-family: ui-monospace, Menlo, Consolas, monospace; '
        'font-size:15px; word-wrap:break-word; white-space:pre-wrap;">'
        + "".join(pieces)
        + "</div>"
    )


def color_legend_html() -> str:
    stops = [f"{prob_to_color(i / 10)} {i*10:.0f}%" for i in range(11)]
    gradient = ", ".join(stops)
    return (
        f'<div style="height:14px; border-radius:4px; '
        f'background: linear-gradient(to right, {gradient});"></div>'
        '<div style="display:flex; justify-content:space-between; '
        'font-size:12px; color:#666; margin-top:2px;">'
        '<span>0.0 — human</span><span>0.5</span><span>1.0 — AI</span></div>'
    )


@st.cache_resource(show_spinner=False)
def _load_model():
    """Load the configured model. Cached across reruns."""
    if MODEL_TYPE == "baseline":
        from inference import load_baseline
        model, tokenizer = load_baseline(
            CHECKPOINT_PATH, BASELINE_MODEL_SIZE, BASELINE_TOKENIZER, DEVICE
        )
        meta = {
            "type": "baseline",
            "size": BASELINE_MODEL_SIZE,
            "tokenizer": BASELINE_TOKENIZER,
            "checkpoint": CHECKPOINT_PATH,
        }
        return model, tokenizer, meta
    if MODEL_TYPE == "finetune":
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
        return model, tokenizer, meta
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE!r}")


@st.cache_resource(show_spinner=False)
def _bootstrap():
    """Validate config, load the model, run a sanity inference. Cached."""
    if MODEL_TYPE not in ("baseline", "finetune"):
        raise ValueError(
            f"MODEL_TYPE must be 'baseline' or 'finetune', got {MODEL_TYPE!r}."
        )
    if DEVICE not in ("cpu", "cuda"):
        raise ValueError(f"DEVICE must be 'cpu' or 'cuda', got {DEVICE!r}.")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"CHECKPOINT_PATH does not exist: {CHECKPOINT_PATH}"
        )
    if MODEL_TYPE == "finetune":
        base_path, _, _ = _resolve_finetune_paths()
        if not os.path.isdir(base_path):
            raise FileNotFoundError(
                f"Base model directory not found: {base_path}"
            )

    model, tokenizer, meta = _load_model()

    from inference import predict_token_probs
    sanity = predict_token_probs(
        text="The quick brown fox jumps over the lazy dog.",
        model=model,
        tokenizer=tokenizer,
        model_type=MODEL_TYPE,
        device=DEVICE,
    )
    if not sanity:
        raise RuntimeError("Sanity inference produced no tokens.")
    meta["sanity_token_count"] = sum(1 for p in sanity if not p.is_special)
    return model, tokenizer, meta


def main() -> None:
    st.set_page_config(page_title="GenTextDetect", layout="wide")
    st.title("GenTextDetect")
    st.caption(
        "Token-level AI-generated text detection. Hover any token to see "
        "its predicted P(AI)."
    )

    status = st.status("Loading model…", expanded=True)
    try:
        with status:
            st.write(
                f"MODEL_TYPE = `{MODEL_TYPE}` · DEVICE = `{DEVICE}` · "
                f"checkpoint = `{CHECKPOINT_PATH}`"
            )
            st.write("Importing torch + transformers and loading weights…")
            model, tokenizer, meta = _bootstrap()
            st.write(
                f"Sanity inference OK — "
                f"{meta['sanity_token_count']} tokens scored."
            )
        status.update(
            label=f"Model ready · {meta['type']}",
            state="complete",
            expanded=False,
        )
    except Exception as exc:  # noqa: BLE001
        status.update(
            label="Model failed to load",
            state="error",
            expanded=True,
        )
        st.error(f"{type(exc).__name__}: {exc}")
        st.info(
            "Fix the CONFIG block at the top of "
            "`src/gentextdetect/app.py` and rerun."
        )
        st.stop()

    with st.expander("Model details"):
        st.json(meta)

    st.markdown(color_legend_html(), unsafe_allow_html=True)

    tab_text, tab_pdf = st.tabs(["Text", "PDF"])

    with tab_text:
        _text_tab(model, tokenizer)
    with tab_pdf:
        _pdf_tab(model, tokenizer)


def _text_tab(model, tokenizer) -> None:
    text = st.text_area(
        "Text to analyze",
        height=260,
        placeholder="Paste any passage here…",
        key="text_input",
    )
    run = st.button(
        "Analyze", type="primary", use_container_width=False, key="text_run"
    )
    if not run:
        return
    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return

    with st.spinner("Scoring tokens…"):
        from inference import aggregate_probability, predict_token_probs
        predictions = predict_token_probs(
            text=text,
            model=model,
            tokenizer=tokenizer,
            model_type=MODEL_TYPE,
            device=DEVICE,
        )

    if not predictions:
        st.warning("No tokens produced for the given input.")
        return

    overall = aggregate_probability(predictions)
    verdict = (
        "Likely AI-generated" if overall >= 0.5 else "Likely human-authored"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean P(AI)", f"{overall:.3f}")
    m2.metric(
        "Tokens scored",
        str(sum(1 for p in predictions if not p.is_special)),
    )
    m3.metric("Verdict", verdict)

    st.subheader("Token-level highlight")
    st.markdown(render_tokens_html(predictions), unsafe_allow_html=True)

    with st.expander("Per-token table"):
        rows = [
            {
                "token": p.text,
                "p(AI)": round(p.prob, 4),
                "special": p.is_special,
            }
            for p in predictions
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _pdf_tab(model, tokenizer) -> None:
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_upload",
        help=(
            "The PDF text is extracted with per-character bounding boxes, "
            "scored token-by-token, and overlaid with translucent colored "
            "rectangles. Long PDFs are chunked at the model's max length."
        ),
    )
    if uploaded is None:
        return

    if not getattr(tokenizer, "is_fast", False):
        st.error(
            "PDF mode requires a fast tokenizer (one that returns "
            "offset_mapping). The current tokenizer does not."
        )
        return

    run = st.button(
        "Analyze PDF", type="primary", key="pdf_run"
    )
    if not run:
        return

    pdf_bytes = uploaded.getvalue()

    with st.spinner("Extracting text + char bboxes from PDF…"):
        try:
            from pdf_utils import extract_chars
        except ImportError as exc:
            st.error(
                f"PyMuPDF is not installed: {exc}. Run "
                "`pip install pymupdf`."
            )
            return
        text, char_spans = extract_chars(pdf_bytes)

    if not text.strip():
        st.warning(
            "No extractable text found in this PDF (image-only scan?)."
        )
        return

    n_chars = sum(1 for c in char_spans if c is not None)
    st.caption(
        f"Extracted {n_chars:,} characters from {uploaded.name} — "
        "scoring now."
    )

    progress = st.progress(0.0, text="Scoring tokens…")

    def _cb(done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0))

    with st.spinner("Running model on extracted text…"):
        from inference import predict_long_text
        token_probs = predict_long_text(
            text=text,
            model=model,
            tokenizer=tokenizer,
            model_type=MODEL_TYPE,
            device=DEVICE,
            progress_cb=_cb,
        )
    progress.empty()

    if not token_probs:
        st.warning("Tokenizer produced no tokens.")
        return

    valid = [t.prob for t in token_probs if t.char_end > t.char_start]
    overall = sum(valid) / max(len(valid), 1)
    verdict = (
        "Likely AI-generated" if overall >= 0.5 else "Likely human-authored"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean P(AI)", f"{overall:.3f}")
    m2.metric("Tokens scored", f"{len(valid):,}")
    m3.metric("Verdict", verdict)

    with st.spinner("Annotating PDF…"):
        from pdf_utils import annotate_pdf
        annotated = annotate_pdf(
            pdf_bytes=pdf_bytes,
            char_spans=char_spans,
            token_spans=[
                (t.char_start, t.char_end, t.prob) for t in token_probs
            ],
            color_fn=prob_to_rgb,
        )

    out_name = uploaded.name.rsplit(".", 1)[0] + "_annotated.pdf"
    st.download_button(
        "Download annotated PDF",
        data=annotated,
        file_name=out_name,
        mime="application/pdf",
        type="primary",
    )

    if len(annotated) <= 25 * 1024 * 1024:
        with st.expander("Inline preview", expanded=True):
            import base64
            b64 = base64.b64encode(annotated).decode("ascii")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                'width="100%" height="800" '
                'style="border:1px solid #ccc; border-radius:4px;"></iframe>',
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "Annotated PDF exceeds 25 MB — skipping inline preview, "
            "use the download button instead."
        )


if __name__ == "__main__":
    main()
