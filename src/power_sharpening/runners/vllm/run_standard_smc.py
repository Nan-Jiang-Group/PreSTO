"""Compatibility entry point for the patched-vLLM Power-SMC runner.

Run existing commands with:
    python -m power_sharpening.runners.vllm.run_standard_smc --help

The stock-vLLM SMC implementation has been removed. This command now uses the same engine, sampler, CLI, and result
metadata as ``run_custom_smc``. Use ``python -m power_sharpening.runners.vllm.run_custom_smc`` for new runs.
"""

from power_sharpening.runners.vllm.run_custom_smc import (
    REQUIRED_SMC_KEYS,
    build_parser,
    main,
    run_custom_smc as run_smc,
)


if __name__ == "__main__":
    main()
