# Power Sharpening Source Packages

Installable sampler library for inference from sharpened LLM distributions
(`p^alpha`), with Metropolis-Hastings, subtree-prefetching, Power-SMC, multi-try
MH, and Calderhead-style variants.

| package | purpose |
| --- | --- |
| `power_sharpening.common` | Temperature schedules, proposal tree, sampling stats. |
| `power_sharpening.backends.hf` | HuggingFace wrapper (`wrapper.py`) + low-temperature and MH samplers (`samplers/`). |
| `power_sharpening.backends.vllm` | vLLM wrapper, samplers (MH, SMC, standard), and the patched engine (`engine_patch/`). |
| `power_sharpening.backends.sglang` | SGLang wrapper + samplers using the same MH algorithms. |
| `power_sharpening.tasks` | Benchmark loading, prompting, and grading behind a common interface. |
| `power_sharpening.config` | YAML run configs (`dataset.yaml`, `algorithms.yaml`) + loader and context-window helpers. |
| `power_sharpening.runners` | CLI entry points per backend: `python -m power_sharpening.runners.<backend>.<runner>`. |

## Installation

From this `src/` directory:

```bash
pip install -e .           # core only
pip install -e '.[hf]'     # + HuggingFace backend
pip install -e '.[vllm]'   # + vLLM backend
pip install -e '.[sglang]' # + SGLang backend
pip install -e '.[all]'    # everything
```

Backend dependencies are optional so lightweight installs skip the serving stacks.

## Backends

**HuggingFace** (`backends.hf`) reads the proposal stream `log q(x_t | x_<t)` and the
target stream `alpha * log p(x_t | x_<t)` from HuggingFace generation outputs.

**vLLM** (`backends.vllm`) patches the vLLM generation path and adds an `alpha`
sampling parameter:

```python
from power_sharpening.backends.vllm.wrapper import vLLM_Wrapper
from power_sharpening.backends.vllm.engine_patch import SamplingParams

mh_llm = vLLM_Wrapper(model="Qwen/Qwen2.5-Math-7B", engine_type="custom")
outputs = mh_llm.generate(
    "What is 1234 + 5678?",
    sampling_params=SamplingParams(temperature=0.25, alpha=0.4),
)
```

**SGLang** (`backends.sglang`) exposes one log-prob stream per request, so
low-temperature proposal sampling uses two passes: decode at `T = 1 / alpha` for
`log q`, then score the sequence at `T = 1` for `log p` (scaled by `alpha`).

```python
from power_sharpening.backends.sglang.samplers import (
    SGL_LLM_Wrapper,
    sglang_load_model_and_tokenizer,
)

alpha = 4.0
engine, tokenizer = sglang_load_model_and_tokenizer("Qwen/Qwen2.5-Math-7B")
wrapper = SGL_LLM_Wrapper(engine, tokenizer, temperature=1.0 / alpha, alpha=alpha)
```

Shared pieces (`SamplingStats`, temperature schedules, the subtree proposal
tree) live in `power_sharpening.common`.

## Benchmarks — `power_sharpening.tasks`

Shared evaluation and grading utilities for the power-sharpening benchmarks
(MATH, AIME, GPQA, HumanEval, MBPP, LiveCodeBench, MMLU, SWE-bench,
ARC-AGI-2). Each `*_benchmark.py` module owns dataset loading, prompt
formatting, completion extraction, and grading for one benchmark, behind a
common `get_question_and_answer` / `evaluate_completions` interface. It installs as part
of the `power-sharpening` distribution — there is no separate package.

`constants.py` holds the `MODEL_MAP` alias → checkpoint mapping, `MAX_NEW_TOKENS`,
and `model_uses_chat_template`.

### `registry.py` — task registry and shared runner configuration

`registry.py` is the single source of truth for the wiring every experiment runner
needs, across **both** the HuggingFace and vLLM backends. Runners import from it
instead of keeping per-directory `utils.py` copies:

| symbol | purpose |
| --- | --- |
| `TASKS` | task-name → benchmark-class registry (also the `--task` choices). |
| `build_benchmark(task, args, model_str)` | instantiate the selected benchmark with its dataset options. |
| `add_benchmark_selection_args(parser)` | register the common `--task` / `--batch_size` flags. |
| `add_benchmark_args(parser, task)` | register **only** the chosen task's dataset flags (aime/lcb), under a titled help group. |
| `resolve_task(argv=None)` | pre-parse `--task` before the task-specific flags are registered. |
| `set_random_seed(seed)` | seed Python / NumPy / Torch / CUDA. |

Context-length / generation-budget helpers now live in
`power_sharpening.config.model_context`.

#### The argument pattern

`resolve_task` + `add_benchmark_args` implement a two-pass parse so that a task's
dataset-specific flags exist only when that task is chosen (e.g. `--difficulty`
is registered for `--task lcb` and rejected otherwise), and `--help` lists
exactly the relevant flags:

```python
import argparse
from power_sharpening.tasks.registry import (
    add_benchmark_args, add_benchmark_selection_args,
    build_benchmark, resolve_task, set_random_seed,
)

task = resolve_task()                  # throwaway pre-parser (add_help=False)
parser = argparse.ArgumentParser()
add_benchmark_selection_args(parser)   # --task / --batch_size
# ... runner-specific args (--algorithm, --alpha, --mcmc_steps, ...) ...
add_benchmark_args(parser, task)       # only the chosen task's flags
args = parser.parse_args()

set_random_seed(args.seed)
benchmark = build_benchmark(args.task, args, model_str)
```

The vLLM runners keep their own `build_parser(task)` with a required, grouped
`--task` and call `add_benchmark_args(parser, task)` at the end — the selection
flags stay bespoke there, but the task-conditional dataset flags come from here.

#### Adding a benchmark

1. Add a `*_benchmark.py` module implementing the `get_question_and_answer` /
   `evaluate_completions` interface.
2. Register it in `TASKS`, and add its construction branch to `build_benchmark`.
3. If it has dataset-specific CLI flags, register them in `add_benchmark_args`
   (keyed on the task) so `build_benchmark` can read them.

`registry.py` uses relative submodule imports (`from .aime_benchmark import ...`),
so it never round-trips through `power_sharpening/tasks/__init__` — no circular
import.
