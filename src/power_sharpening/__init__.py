"""Samplers for sharpened LLM distributions (p^alpha).

Layout: ``common`` (schedules, proposal tree, stats), ``backends`` (hf /
vllm / sglang wrappers + samplers), ``tasks`` (benchmarks + grading), ``config`` (YAML run configs), ``runners`` (CLI
entry points, run with ``python -m power_sharpening.runners.<backend>.<runner>``).

Install with ``pip install -e ./src`` from the repository root.
"""
