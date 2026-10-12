#!/usr/bin/env python3
"""Plot mean MH transitions and directly logged MH-step time.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_mean_mh_transitions_and_empirical_time.py \
        --directory /path/to/vllm/logs
"""

from __future__ import annotations

from case_studies.draw.figures.endpoints.endpoint_metrics import (
    EndpointFigureSpec,
    EndpointPanel,
    run_endpoint_figure_cli,
)

FIGURE_SUFFIX = ".mean-mh-transitions-and-empirical-time.pdf"
PANELS = (
    EndpointPanel(
        "mean_mh_transitions_per_target_model_call",
        "mean MH transitions per call",
        None,
        dynamic_y_step=0.5,
        dynamic_y_padding=0.25,
        dynamic_y_min=1.0,
    ),
    EndpointPanel(
        "mean_logged_mh_step_seconds",
        "mean time per call (sec)",
        None,
        dynamic_y_step=1.0,
        dynamic_y_padding=0.5,
        dynamic_y_min=3.0,
    ),
)
SPEC = EndpointFigureSpec(
    figure_suffix=FIGURE_SUFFIX,
    panels=PANELS,
    description=(
        "Plot mean MH transitions per call and direct mean time from logged "
        "MH-step durations across compatible prefetch budgets."
    ),
)


if __name__ == "__main__":
    run_endpoint_figure_cli(SPEC)
