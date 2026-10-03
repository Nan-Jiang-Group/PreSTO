"""Backend-agnostic utilities: temperature schedules, proposal tree, stats.

Install the package with ``pip install -e ./src`` from the repository root. Backend-specific model wrappers live in
``power_sharpening.backends``.
"""

from .temp_scheduler import (
    SCHEDULE_TYPES,
    ConstantScheduler,
    CosineScheduler,
    CosineWarmRestartsScheduler,
    CyclicScheduler,
    ExponentialDecayScheduler,
    LinearScheduler,
    StepScheduler,
    TemperatureScheduler,
    set_schedule,
)
from .sample_stats import SamplingStats


__all__ = [
    "SCHEDULE_TYPES",
    "ConstantScheduler",
    "CosineScheduler",
    "CosineWarmRestartsScheduler",
    "CyclicScheduler",
    "ExponentialDecayScheduler",
    "LinearScheduler",
    "StepScheduler",
    "TemperatureScheduler",
    "set_schedule",
    "SamplingStats",
]
