"""PDF text extraction and per-token color overlay using PyMuPDF.

Each PDF character is preserved alongside its (page, bbox), and text from all
pages is concatenated into a single string. After running per-token inference
on that string, every token's char-offset span is mapped back to one or more
bboxes and a colored translucent rectangle is drawn over those bboxes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class CharSpan:
    page_idx: int
    bbox: Tuple[float, float, float, float]


def extract_chars(pdf_bytes: bytes) -> Tuple[str, List[Optional[CharSpan]]]:
    """Return ``(concat_text, char_spans)`` where the two are index-aligned.

    For each real character extracted from the PDF the corresponding entry in
    ``char_spans`` is a ``CharSpan``. Synthetic separators (line breaks, page
    breaks) appear in ``concat_text`` for readability but their entries are
    ``None`` so they cannot be highlighted.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pieces: List[str] = []
    spans: List[Optional[CharSpan]] = []

    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("rawdict")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        c = ch.get("c", "")
                        if not c:
                            continue
                        pieces.append(c)
                        spans.append(
                            CharSpan(
                                page_idx=page_idx,
                                bbox=tuple(ch["bbox"]),  # type: ignore[arg-type]
                            )
                        )
                pieces.append("\n")
                spans.append(None)
            pieces.append("\n")
            spans.append(None)
        pieces.append("\n")
        spans.append(None)

    doc.close()
    return "".join(pieces), spans


def _group_bboxes_by_line(
    bboxes: List[Tuple[float, float, float, float]],
    y_tol: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """Merge bboxes that share roughly the same vertical band into single
    rectangles, so a multi-character token becomes one highlight per line."""
    if not bboxes:
        return []
    sorted_bb = sorted(bboxes, key=lambda b: (round(b[1], 1), b[0]))
    merged: List[List[float]] = [list(sorted_bb[0])]
    for b in sorted_bb[1:]:
        cur = merged[-1]
        same_line = abs(b[1] - cur[1]) < y_tol and abs(b[3] - cur[3]) < y_tol
        if same_line:
            cur[0] = min(cur[0], b[0])
            cur[2] = max(cur[2], b[2])
            cur[1] = min(cur[1], b[1])
            cur[3] = max(cur[3], b[3])
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]  # type: ignore[return-value]


def annotate_pdf(
    pdf_bytes: bytes,
    char_spans: List[Optional[CharSpan]],
    token_spans: List[Tuple[int, int, float]],
    color_fn: Callable[[float], Tuple[float, float, float]],
    fill_opacity: float = 0.45,
) -> bytes:
    """Overlay a translucent colored rectangle on the bboxes of each token.

    ``token_spans`` is a list of ``(char_start, char_end, p_ai)`` produced by
    ``predict_long_text``. ``color_fn`` maps a probability in [0, 1] to an
    ``(r, g, b)`` tuple of floats in [0, 1] (PyMuPDF's color convention).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for char_start, char_end, prob in token_spans:
        if char_end <= char_start:
            continue  # special tokens / empty spans

        per_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for i in range(char_start, char_end):
            if i >= len(char_spans):
                break
            cs = char_spans[i]
            if cs is None:
                continue
            per_page.setdefault(cs.page_idx, []).append(cs.bbox)

        if not per_page:
            continue

        rgb = color_fn(prob)
        for page_idx, bboxes in per_page.items():
            page = doc[page_idx]
            for line_bbox in _group_bboxes_by_line(bboxes):
                rect = fitz.Rect(*line_bbox)
                page.draw_rect(
                    rect,
                    color=None,
                    fill=rgb,
                    fill_opacity=fill_opacity,
                    overlay=True,
                )

    out = io.BytesIO()
    doc.save(out, garbage=3, deflate=True)
    doc.close()
    return out.getvalue()
