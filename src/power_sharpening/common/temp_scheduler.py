"""Temperature schedule implementations.

Run repository scripts with:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python <script>
"""

import math

SCHEDULE_TYPES = [
    "const",
    "cosine",
    "step",
    "exp_decay",
    "cyclic",
    "linear",
    "cosine_warm_restarts",
]

class TemperatureScheduler:
    """Base class for temperature schedulers (analogous to PyTorch LRScheduler)."""

    is_anneal = True

    def __init__(self, initial_temp: float, **kwargs):
        self.initial_temp = initial_temp
        self.step_count = 0

    def step(self) -> float:
        """Advance one step and return the new temperature."""
        raise NotImplementedError

    def get_temperature(self) -> float:
        raise NotImplementedError


class ConstantScheduler(TemperatureScheduler):
    """Temperature stays fixed."""

    is_anneal = False

    def step(self) -> float:
        self.step_count += 1
        return self.initial_temp

    def get_temperature(self) -> float:
        return self.initial_temp


class CosineScheduler(TemperatureScheduler):
    """Cosine annealing from ``initial_temp`` to ``min_temp`` over ``total_steps``.

        T_t = min_temp + 0.5 * (initial_temp - min_temp) * (1 + cos(pi * t / total_steps))
    """

    def __init__(self, initial_temp: float, min_temp: float = 0.1, total_steps: int = 100, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.min_temp = min_temp
        self.total_steps = total_steps

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        t = min(self.step_count, self.total_steps)
        return self.min_temp + 0.5 * (self.initial_temp - self.min_temp) * (
                1 + math.cos(math.pi * t / self.total_steps)
        )


class StepScheduler(TemperatureScheduler):
    """Two-phase step schedule: high temperature for the first 70% of steps,
    then low temperature for the remaining 30%.

        T_t = initial_temp                          if t < 0.7 * total_steps
              initial_temp * gamma                   otherwise
    """

    def __init__(self, initial_temp: float, total_steps: int, gamma: float = 0.5, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.total_steps = total_steps
        self.switch_step = int(0.7 * total_steps)
        self.gamma = gamma

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        if self.step_count < self.switch_step:
            return self.initial_temp
        return self.initial_temp * self.gamma


class ExponentialDecayScheduler(TemperatureScheduler):
    """Multiply temperature by ``gamma`` every step.

        T_t = initial_temp * gamma ^ t
    """

    def __init__(self, initial_temp: float, gamma: float = 0.99, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.gamma = gamma

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        return self.initial_temp * (self.gamma ** self.step_count)


class CyclicScheduler(TemperatureScheduler):
    """Triangular cyclic schedule between ``min_temp`` and ``max_temp``.

    Each cycle spans ``2 * step_size_up`` steps (linear ramp up then down).

    Supports three modes (matching CyclicLR):
        * ``"triangular"``  — fixed amplitude each cycle.
        * ``"triangular2"`` — amplitude halved every cycle.
        * ``"exp_range"``   — amplitude scaled by ``gamma ^ t`` each step.
    """

    def __init__(self, initial_temp: float, min_temp: float, max_temp: float,
                 step_size_up: int = 50, mode: str = "triangular", gamma: float = 1.0, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.step_size_up = step_size_up
        self.mode = mode
        self.gamma = gamma

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        cycle_len = 2 * self.step_size_up
        cycle = self.step_count // cycle_len
        x = self.step_count - cycle * cycle_len
        # x in [0, cycle_len): ramp up for step_size_up, ramp down for step_size_up
        if x < self.step_size_up:
            frac = x / self.step_size_up  # 0 -> 1
        else:
            frac = (cycle_len - x) / self.step_size_up  # 1 -> 0

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** cycle)
        elif self.mode == "exp_range":
            scale = self.gamma ** self.step_count
        else:
            raise ValueError(f"Unknown cyclic mode: {self.mode!r}")

        return self.min_temp + (self.max_temp - self.min_temp) * frac * scale

class LinearScheduler(TemperatureScheduler):
    """Linear interpolation from ``initial_temp`` to ``end_temp`` over ``total_steps``.

        T_t = initial_temp + (end_temp - initial_temp) * min(t / total_steps, 1)
    """

    def __init__(self, initial_temp: float, end_temp: float = 0.1, total_steps: int = 100, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.end_temp = end_temp
        self.total_steps = total_steps

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        t = min(self.step_count, self.total_steps)
        return self.initial_temp + (self.end_temp - self.initial_temp) * (t / self.total_steps)


class CosineWarmRestartsScheduler(TemperatureScheduler):
    """Cosine annealing with periodic warm restarts.

    Every ``T_0`` steps the temperature resets to ``initial_temp`` and the cycle length is multiplied by ``T_mult``.

        T_t = min_temp + 0.5 * (initial_temp - min_temp) * (1 + cos(pi * T_cur / T_i))
    """

    def __init__(self, initial_temp: float, min_temp: float = 0.1,
                 T_0: int = 50, T_mult: int = 1, **kwargs):
        super().__init__(initial_temp, **kwargs)
        self.min_temp = min_temp
        self.T_0 = T_0
        self.T_mult = T_mult

    def step(self) -> float:
        self.step_count += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        # find which cycle we are in and how far through it
        T_cur = self.step_count
        T_i = self.T_0
        if self.T_mult == 1:
            T_cur = T_cur % T_i
        else:
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult
        return self.min_temp + 0.5 * (self.initial_temp - self.min_temp) * (
                1 + math.cos(math.pi * T_cur / T_i)
        )

def set_schedule(initial_temp: float, temperature_schedule_type: str = "const",
                 **kwargs) -> TemperatureScheduler:
    """Factory that returns the appropriate temperature scheduler.

    Args:
        initial_temp: starting temperature.
        temperature_schedule_type: one of ``"const"``, ``"cosine"``, ``"step"``, ``"exp_decay"``, ``"cyclic"``,
            ``"linear"``, ``"cosine_warm_restarts"``.
        **kwargs: forwarded to the chosen scheduler's ``__init__``.

    Returns:
        A ``TemperatureScheduler`` instance.
    """
    schedulers = {
        "const": ConstantScheduler,
        "cosine": CosineScheduler,
        "step": StepScheduler,
        "exp_decay": ExponentialDecayScheduler,
        "cyclic": CyclicScheduler,
        "linear": LinearScheduler,
        "cosine_warm_restarts": CosineWarmRestartsScheduler,
    }
    if temperature_schedule_type not in schedulers:
        valid = ", ".join(f"'{k}'" for k in schedulers)
        raise ValueError(f"Unknown schedule type: {temperature_schedule_type!r}. Choose from {valid}.")

    cls = schedulers[temperature_schedule_type]
    return cls(initial_temp, **kwargs)
