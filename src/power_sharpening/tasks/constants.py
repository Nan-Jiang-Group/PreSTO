"""
Single source of truth for prompt templates and model configuration.

Import MODEL_MAP or model_uses_chat_template from power_sharpening.tasks.constants;
this module has no standalone CLI.

Previously duplicated in:
  - baselines/power-sharpening-hf/constants.py
  - subtree-prefetching-sampler/constants.py
"""

from transformers import AutoConfig



MODEL_MAP = {
    # qwen2.5
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen-math-small": "Qwen/Qwen2.5-Math-1.5B",
    "qwen-math-medium": "Qwen/Qwen2.5-Math-7B",
    "qwen-math-grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "qwen-grpo": "fhai50032/Qwen2.5-GRPO-7B",
    "qwen-coder": "Qwen/Qwen2.5-Coder-7B",
    "qwen-coder-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    # qwen3
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-4b-grpo": "shjondhale/AzureML-Qwen3-4B-Base-GRPO",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
    "qwen3-8b-grpo": "jadohu/Qwen3-8B-GRPO",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    #
    "qwen3.5-4b":"Qwen/Qwen3.5-4B",
    "qwen3.5-9b":"Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    # gemma
    "gemma-12b-it": "google/gemma-4-12B-it",
    # deepseek
    "deepseek-math": "deepseek-ai/deepseek-math-7b-base",
    "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-base",
    "deepseek-math-instruct": "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-math-grpo": "deepseek-ai/deepseek-math-7b-rl",
    # openai
    "gpt-oss-20b": "openai/gpt-oss-20b",
    # phi
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "phi4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    # llama / tulu
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b-grpo": "parkjo/Llama-3.1-8B-Instruct_grpo_ppl_adv_rollout_8_kl_0.001_20260516_140637_step232",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "tulu3-grpo": "allenai/Llama-3.1-Tulu-3-8B",
    "tulu3-sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "tulu3-dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
}

# Model aliases whose checkpoints expect conversation-formatted input. Aliases not listed here receive the raw benchmark
# prompt.
CHAT_TEMPLATE_MODELS = frozenset({
    "qwen-coder-instruct",
    "qwen2.5-coder-7b-instruct",
    "qwen3-4b",
    "qwen3-4b-instruct",
    "qwen3-4b-thinking",
    "qwen3-8b",
    "qwen3-8b-thinking",
    "qwen3-8b-grpo",
    "qwen3-30b-a3b",
    "qwen3.5-27b",
    "gemma-12b-it",
    "deepseek-math-instruct",
    "deepseek-math-grpo",
    "gpt-oss-20b",
    "phi",
    "phi3.5",
    "phi4-mini-instruct",
    "llama3.1-8b-instruct",
    "llama3.1-8b-grpo",
    "tulu",
    "tulu3-grpo",
    "tulu3-sft",
    "tulu3-dpo",
})


def model_uses_chat_template(model_name: str) -> bool:
    if model_name not in MODEL_MAP:
        raise KeyError(f"Unknown model alias: {model_name!r}")
    return model_name in CHAT_TEMPLATE_MODELS



MAX_NEW_TOKENS = 3072
