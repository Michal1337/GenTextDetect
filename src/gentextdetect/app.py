"""Streamlit UI for token-level AI-generated text detection.

Run with::

    streamlit run src/gentextdetect/app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st
import torch

from inference import (
    BASELINE_MODELS,
    TokenPrediction,
    aggregate_probability,
    load_baseline,
    load_finetune,
    predict_token_probs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE_DIR = REPO_ROOT / "checkpoints" / "baseline"
DEFAULT_FINETUNE_DIR = REPO_ROOT / "checkpoints" / "finetuned"


def prob_to_color(prob: float) -> str:
    """Green (low prob = human) → yellow → red (high prob = AI)."""
    prob = max(0.0, min(1.0, prob))
    if prob < 0.5:
        # green → yellow
        t = prob / 0.5
        r = int(80 + (255 - 80) * t)
        g = 200
    else:
        # yellow → red
        t = (prob - 0.5) / 0.5
        r = 255
        g = int(200 - (200 - 60) * t)
    b = 60
    return f"rgba({r}, {g}, {b}, 0.55)"


def render_tokens_html(predictions: list[TokenPrediction]) -> str:
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
    stops = []
    for i in range(11):
        p = i / 10
        stops.append(f"{prob_to_color(p)} {p*100:.0f}%")
    gradient = ", ".join(stops)
    return (
        f'<div style="height:14px; border-radius:4px; '
        f'background: linear-gradient(to right, {gradient});"></div>'
        '<div style="display:flex; justify-content:space-between; '
        'font-size:12px; color:#666; margin-top:2px;">'
        '<span>0.0 — human</span><span>0.5</span><span>1.0 — AI</span></div>'
    )


@st.cache_resource(show_spinner=False)
def _cached_baseline(checkpoint_path: str, model_size: str, tokenizer_name: str, device: str):
    return load_baseline(checkpoint_path, model_size, tokenizer_name, device)


@st.cache_resource(show_spinner=False)
def _cached_finetune(
    checkpoint_path: str,
    base_model_path: str,
    base_model_name: str,
    is_phi: bool,
    device: str,
):
    return load_finetune(
        checkpoint_path,
        base_model_path,
        base_model_name or None,
        is_phi=is_phi,
        device=device,
    )


def list_checkpoints(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(str(p) for p in folder.glob("*.pt"))


def sidebar_config() -> dict:
    st.sidebar.header("Model")

    model_type = st.sidebar.selectbox(
        "Model type",
        ["baseline", "finetune", "finetune (phi)"],
        index=0,
        help=(
            "`baseline` is a small transformer trained from scratch — runs on "
            "CPU. `finetune` variants attach a classifier head onto a base "
            "LLM and need a CUDA GPU with flash-attn."
        ),
    )

    cuda_ok = torch.cuda.is_available()
    device_choices = ["cpu"] + (["cuda"] if cuda_ok else [])
    default_device = "cuda" if cuda_ok and model_type != "baseline" else "cpu"
    device = st.sidebar.selectbox(
        "Device",
        device_choices,
        index=device_choices.index(default_device),
    )

    cfg: dict = {"model_type": model_type, "device": device}

    if model_type == "baseline":
        cfg["model_size"] = st.sidebar.selectbox(
            "Baseline size", list(BASELINE_MODELS), index=0
        )
        cfg["tokenizer_name"] = st.sidebar.text_input(
            "Tokenizer (HF name or local path)",
            value="meta-llama/Llama-3.2-1B-Instruct",
        )
        suggested = list_checkpoints(DEFAULT_BASELINE_DIR)
        cfg["checkpoint_path"] = st.sidebar.selectbox(
            "Checkpoint (.pt)",
            options=suggested + ["<custom path>"] if suggested else ["<custom path>"],
            index=0,
        )
        if cfg["checkpoint_path"] == "<custom path>":
            cfg["checkpoint_path"] = st.sidebar.text_input(
                "Custom checkpoint path", value=""
            )
    else:
        suggested = list_checkpoints(DEFAULT_FINETUNE_DIR)
        cfg["checkpoint_path"] = st.sidebar.selectbox(
            "Classifier head (.pt)",
            options=suggested + ["<custom path>"] if suggested else ["<custom path>"],
            index=0,
        )
        if cfg["checkpoint_path"] == "<custom path>":
            cfg["checkpoint_path"] = st.sidebar.text_input(
                "Custom checkpoint path", value=""
            )
        cfg["base_model_path"] = st.sidebar.text_input(
            "Base model directory",
            value="",
            help="Local path to the base LLM weights (HF format).",
        )
        cfg["base_model_name"] = st.sidebar.text_input(
            "Base model name (for PAD_TOKENS lookup)",
            value="",
            help="Optional. e.g. Llama-3.2-3B-Instruct, Mistral-Nemo-Instruct-2407.",
        )

    return cfg


def load_model(cfg: dict):
    if cfg["model_type"] == "baseline":
        return _cached_baseline(
            cfg["checkpoint_path"],
            cfg["model_size"],
            cfg["tokenizer_name"],
            cfg["device"],
        )
    return _cached_finetune(
        cfg["checkpoint_path"],
        cfg["base_model_path"],
        cfg["base_model_name"],
        is_phi=cfg["model_type"] == "finetune (phi)",
        device=cfg["device"],
    )


def main() -> None:
    st.set_page_config(
        page_title="GenTextDetect",
        page_icon=None,
        layout="wide",
    )
    st.title("GenTextDetect")
    st.caption(
        "Token-level AI-generated text detection. Hover any token to see "
        "its predicted P(AI)."
    )

    cfg = sidebar_config()

    text = st.text_area(
        "Text to analyze",
        height=260,
        placeholder="Paste any passage here…",
    )

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run = st.button("Analyze", type="primary", use_container_width=True)
    with col_info:
        st.markdown(color_legend_html(), unsafe_allow_html=True)

    if not run:
        return

    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return

    if not cfg.get("checkpoint_path") or not os.path.isfile(
        cfg["checkpoint_path"]
    ):
        st.error(
            "Checkpoint path is missing or does not exist. Set it in the "
            "sidebar."
        )
        return

    try:
        with st.spinner("Loading model…"):
            model, tokenizer = load_model(cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model: {exc}")
        return

    model_type_short = "baseline" if cfg["model_type"] == "baseline" else "finetune"

    with st.spinner("Scoring tokens…"):
        predictions = predict_token_probs(
            text=text,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type_short,
            device=cfg["device"],
        )

    if not predictions:
        st.warning("No tokens produced for the given input.")
        return

    overall = aggregate_probability(predictions)
    verdict = "Likely AI-generated" if overall >= 0.5 else "Likely human-authored"

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean P(AI)", f"{overall:.3f}")
    m2.metric("Tokens scored", str(sum(1 for p in predictions if not p.is_special)))
    m3.metric("Verdict", verdict)

    st.subheader("Token-level highlight")
    st.markdown(render_tokens_html(predictions), unsafe_allow_html=True)

    with st.expander("Per-token table"):
        rows = [
            {"token": p.text, "p(AI)": round(p.prob, 4), "special": p.is_special}
            for p in predictions
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
