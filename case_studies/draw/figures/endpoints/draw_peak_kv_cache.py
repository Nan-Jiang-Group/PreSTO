#!/usr/bin/env python3
"""Plot peak KV-cache occupancy across prefetch sizes.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_peak_kv_cache.py \
        --directory /path/to/vllm/logs
"""

from __future__ import annotations

from case_studies.draw.figures.endpoints.endpoint_metrics import (
    EndpointFigureSpec,
    EndpointPanel,
    run_endpoint_figure_cli,
)

FIGURE_SUFFIX = ".peak-kv-cache.pdf"
PANELS = (
    EndpointPanel(
        "peak_kv_cache_occupancy_percent",
        r"peak KV-cache occupancy (\%)",
        # Occupancy depends on the model, the KV budget, and how many runs shared the GPU, so a ceiling fixed for one
        # sweep goes stale on the
        # next: the 2.2 tuned when the largest budgets neared 2 percent left
        # half the panel empty once they topped out near 1.15. The range is taken from the data instead, from zero
        # because these are bars and a lifted baseline would exaggerate the differences between them.
        None,
        dynamic_y_step=0.25,
        dynamic_y_padding=0.05,
        dynamic_y_min=0.0,
    ),
)
SPEC = EndpointFigureSpec(
    figure_suffix=FIGURE_SUFFIX,
    panels=PANELS,
    description=(
        "Plot peak KV-cache occupancy across compatible prefetch budgets."
    ),
    require_resource_summaries=True,
)


if __name__ == "__main__":
    run_endpoint_figure_cli(SPEC)
