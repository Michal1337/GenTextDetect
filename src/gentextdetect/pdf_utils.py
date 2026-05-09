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


def _sort_blocks_for_reading(blocks, page_width: float, full_width_frac: float = 0.6):
    """Order blocks for column-aware reading order.

    Each block's horizontal center is compared against the page midline:
    blocks wider than ``full_width_frac * page_width`` are treated as
    full-width (titles, banners, full-width figures) and sorted purely by
    ``y0`` *before* the columnar blocks. Remaining blocks are bucketed into
    a left column (``col=0``) or right column (``col=1``) and read top-down
    within each.
    """
    if not blocks:
        return []
    page_mid = page_width / 2
    full_thr = full_width_frac * page_width

    def sort_key(b):
        x0, y0, x1, _ = b["bbox"]
        width = x1 - x0
        x_center = (x0 + x1) / 2
        if width > full_thr:
            return (-1, y0, x0)
        return (0 if x_center < page_mid else 1, y0, x0)

    text_blocks = [
        b for b in blocks
        if b.get("type", 0) == 0 and "bbox" in b and "lines" in b
    ]
    return sorted(text_blocks, key=sort_key)


def extract_chars(pdf_bytes: bytes) -> Tuple[str, List[Optional[CharSpan]]]:
    """Return ``(concat_text, char_spans)`` where the two are index-aligned.

    For each real character extracted from the PDF the corresponding entry in
    ``char_spans`` is a ``CharSpan``. Synthetic separators (line breaks, page
    breaks) appear in ``concat_text`` for readability but their entries are
    ``None`` so they cannot be highlighted.

    Blocks are reordered per page for column-aware reading order so that
    two-column layouts produce coherent prose for the model.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pieces: List[str] = []
    spans: List[Optional[CharSpan]] = []

    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("rawdict")
        page_width = float(page.rect.width)
        ordered_blocks = _sort_blocks_for_reading(
            page_dict.get("blocks", []), page_width
        )
        for block in ordered_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        c = ch.get("c", "")
                        if not c:
                            continue
                        bbox = tuple(ch["bbox"])  # type: ignore[arg-type]
                        # Ligatures (e.g. "fi") may arrive as multi-char
                        # strings — split them so the joined text and the
                        # char_spans list stay 1:1 in length.
                        for sub in c:
                            pieces.append(sub)
                            spans.append(
                                CharSpan(page_idx=page_idx, bbox=bbox)
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
