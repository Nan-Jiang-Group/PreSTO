"""CPU tests for EntropyCut YAML defaults and runner validation.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_entropy_cut_config.py
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if (
    "sglang" not in sys.modules
    and importlib.util.find_spec("sglang") is None
):
    sys.modules["sglang"] = types.ModuleType("sglang")

from power_sharpening.config import load_run_config
from power_sharpening.runners.sglang.run_entropy_cut_mh import (
    _validate_and_resolve,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _args(**overrides):
    values = {
        "alpha": 4.0,
        "temperature": -1.0,
        "temperature_schedule_type": "const",
        "mcmc_steps": 10,
        "num_blocks": 16,
        "max_new_tokens": 3072,
        "cut_power": 4.0,
        "entropy_mode": "topk",
        "entropy_top_k": 64,
        "max_samples": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_yaml_defaults_match_entropy_cut_math500_run():
    config = load_run_config(
        REPO_ROOT / "src" / "power_sharpening" / "config",
        "math500",
        "entropy_cut_mh",
    )

    assert config["alpha"] == 4.0
    assert config["temperature"] == -1
    assert config["temperature_schedule_type"] == "const"
    assert config["mcmc_steps"] == 10
    assert config["num_blocks"] == 16
    assert config["cut_power"] == 4.0
    assert config["entropy_mode"] == "topk"
    assert config["entropy_top_k"] == 64
    assert config["max_new_tokens"] == 3072


def test_paper_temperature_is_resolved_from_alpha():
    assert _validate_and_resolve(_args()) == pytest.approx(0.25)
    assert _validate_and_resolve(
        _args(alpha=5.0, temperature=0.2)
    ) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature_schedule_type": "linear"}, "must be 'const'"),
        ({"temperature": 0.5}, "1/alpha"),
        ({"max_new_tokens": 10}, "divisible"),
        ({"cut_power": -1.0}, "cut_power"),
        ({"cut_power": float("nan")}, "cut_power"),
        ({"entropy_top_k": 0}, "positive integer"),
        ({"entropy_top_k": True}, "positive integer"),
        (
            {"entropy_mode": "exact", "entropy_top_k": None},
            "not available",
        ),
        ({"max_samples": 0}, "positive integer or null"),
    ],
)
def test_invalid_or_unsupported_configs_fail_early(overrides, message):
    with pytest.raises(SystemExit, match=message):
        _validate_and_resolve(_args(**overrides))
