"""Print the text that would be passed to the model for a given PDF.

Usage::

    python src/gentextdetect/dump_pdf_text.py path/to/paper.pdf
    python src/gentextdetect/dump_pdf_text.py path/to/paper.pdf --out extracted.txt

Runs the same extraction pipeline as the Streamlit app and the batch
annotator (column-aware reading order, per-char bbox preservation), then
dumps the concatenated text to stdout or a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_utils import extract_chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print extraction stats to stderr.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Drop math-font spans and lines that look like math fragments "
            "or plot labels."
        ),
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    text, char_spans = extract_chars(args.pdf.read_bytes(), clean=args.clean)

    if args.stats:
        n_real = sum(1 for cs in char_spans if cs is not None)
        n_synthetic = len(char_spans) - n_real
        n_pages = (
            max((cs.page_idx for cs in char_spans if cs is not None), default=-1)
            + 1
        )
        print(
            f"[{args.pdf.name}] "
            f"{n_real:,} real chars · {n_synthetic:,} synthetic separators · "
            f"{n_pages} page(s) · {len(text):,} chars total",
            file=sys.stderr,
        )

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text):,} chars to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
