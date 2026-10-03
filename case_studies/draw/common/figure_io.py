"""Write the figures and source-data tables the drawing CLIs produce.

Every figure here is saved the same way -- create the parent directory, write a tight-bounded PDF 1.7, close the figure
even if saving fails -- and every table is a CSV written beside it. PDF export requires the qpdf executable on PATH.
Both live here so a caller states only what it writes.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

Row = dict[str, object]


def ensure_pdf_17(path: Path) -> Path:
    """Rewrite a PDF as version 1.7 and validate it before replacing the file.

    Args:
        path: PDF to update in place; its vector graphics and fonts are retained.

    Returns:
        The same path after qpdf has written and checked the PDF 1.7 file.
    """
    qpdf = shutil.which("qpdf")
    if qpdf is None:
        raise RuntimeError("PDF 1.7 export requires the qpdf executable on PATH")
    path = path.resolve()
    with tempfile.TemporaryDirectory(prefix=".pdf-1.7-", dir=path.parent) as directory:
        converted = Path(directory) / path.name
        subprocess.run(
            [qpdf, "--force-version=1.7", str(path), str(converted)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [qpdf, "--check", str(converted)],
            check=True, capture_output=True, text=True,
        )
        with converted.open("rb") as pdf_file:
            if pdf_file.readline().strip() != b"%PDF-1.7":
                raise RuntimeError(f"Expected a PDF 1.7 header in {converted}")
        converted.replace(path)
    return path


def save_figure(figure: Figure, path: Path, *, pad_inches: float | None = None) -> Path:
    """Save one figure as an editable PDF 1.7 and always close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if pad_inches is None else {"pad_inches": pad_inches}
    try:
        figure.savefig(path, bbox_inches="tight", **options)
        if path.suffix.lower() == ".pdf":
            ensure_pdf_17(path)
    finally:
        plt.close(figure)
    return path


def write_dict_rows(
    path: Path,
    rows: Sequence[Row],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write records to CSV, taking the columns from the first row by default."""
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot write empty source data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames if fieldnames is not None else rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
