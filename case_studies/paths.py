"""Absolute locations shared by the case-study drawing and extraction code.

This library is not run directly; import it from a figure or parser module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

CASE_STUDIES_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = CASE_STUDIES_DIR.parent
DRAW_DIR: Final = CASE_STUDIES_DIR / "draw"
EXTRACT_DIR: Final = CASE_STUDIES_DIR / "extract"

# Every runner writes its logs and per-run artifacts under case_studies/logs/<dataset>/<date>, so the drawing CLIs
# default there rather than beside their own source.
LOGS_DIR: Final = CASE_STUDIES_DIR / "logs"
