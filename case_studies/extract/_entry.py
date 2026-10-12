"""Launch a packaged case-study CLI from its historical script path.

Wrappers in this directory call ``run`` so that ``python case_studies/<draw|extract>/<name>.py`` keeps working while the
code lives in the ``case_studies`` package. This file is a library; run a sibling wrapper instead.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# The directory holding ``case_studies``; a script's own directory is the only thing Python puts on sys.path for it.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def run(module: str) -> None:
    """Execute ``module`` exactly as ``python -m`` would, with the caller's argv."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    runpy.run_module(module, run_name="__main__", alter_sys=True)
