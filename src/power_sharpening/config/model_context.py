"""Context-window helpers for generation runners.

Run a quick import check with:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python -m py_compile /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/model_context.py
"""

from transformers import AutoConfig


def checkpoint_max_model_len(model_name_or_path: str) -> int | None:
    """Read the context length declared by a Hugging Face checkpoint."""
    config = AutoConfig.from_pretrained(model_name_or_path)
    for field in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ):
        value = getattr(config, field, None)
        if isinstance(value, (int, float)) and 0 < value < 10**9:
            return int(value)
    return None


def resolve_max_model_len(
    model_name_or_path: str,
    requested: int | None,
) -> tuple[int | None, int | None]:
    """Cap a requested context length at the checkpoint's declared limit."""
    checkpoint_limit = checkpoint_max_model_len(model_name_or_path)
    if requested is None:
        return checkpoint_limit, checkpoint_limit
    if checkpoint_limit is None:
        return requested, None
    return min(requested, checkpoint_limit), checkpoint_limit


def batch_max_new_tokens(
    tokenizer,
    prompts: list[str],
    max_model_len: int | None,
    requested: int,
    safety_margin: int = 16,
) -> int:
    """Compute one safe generation budget shared by a prompt batch."""
    if max_model_len is None:
        return requested
    longest_prompt = max(
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    )
    allowed = min(requested, max_model_len - longest_prompt - safety_margin)
    if allowed <= 0:
        raise ValueError(
            f"Prompt length {longest_prompt} leaves no generation room within "
            f"max_model_len={max_model_len} (safety_margin={safety_margin})."
        )
    return allowed


__all__ = [
    "checkpoint_max_model_len",
    "resolve_max_model_len",
    "batch_max_new_tokens",
]
