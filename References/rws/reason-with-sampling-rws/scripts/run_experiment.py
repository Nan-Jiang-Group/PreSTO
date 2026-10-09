import argparse

import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

from src.rws.models import ModelWrapper
from src.rws.power_mcmc import power_sampling_mcmc
from src.rws.baselines import run_grpo_single, run_grpo_majority_vote
from src.rws.metrics import exact_match
from src.rws.utils import (
    load_json, load_jsonl, load_yaml,
    save_csv,
    set_seed, get_timestamp, ensure_dir
)


def get_model_names_from_config(config: Dict[str, Any], methods: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract model names from config based on which methods are requested.
    
    Supports two modes:
    1. Legacy mode: `model_name` used for all methods
    2. Dual model mode: `base_model_name` for power_mcmc, `grpo_model_name` for baselines
    
    Args:
        config: Configuration dict
        methods: List of methods to run
        
    Returns:
        Tuple of (base_model_name, grpo_model_name) - either can be None if not needed
    """
    # Check if new dual-model keys are present
    base_model_name = config.get("base_model_name")
    grpo_model_name = config.get("grpo_model_name")
    legacy_model_name = config.get("model_name", config.get("model_id"))

    # Determine which methods need which model
    needs_base_model = "power_mcmc" in methods
    needs_grpo_model = any(m in methods for m in ["grpo_single", "grpo", "grpo_vote", "grpo_majority_vote"])

    # Use dual-model mode if either new key is present
    if base_model_name is not None or grpo_model_name is not None:
        # Dual model mode
        if needs_base_model:
            if base_model_name is None:
                # Fall back to legacy model_name for base model
                base_model_name = legacy_model_name
            if base_model_name is None:
                raise ValueError(
                    "power_mcmc requires base_model_name (or model_name as fallback). "
                    "Please specify in config."
                )

        if needs_grpo_model:
            if grpo_model_name is None:
                # Fall back to legacy model_name for GRPO model
                grpo_model_name = legacy_model_name
            if grpo_model_name is None:
                raise ValueError(
                    "grpo_single/grpo_vote require grpo_model_name (or model_name as fallback). "
                    "Please specify in config."
                )
    else:
        # Legacy mode - use model_name for everything
        if not legacy_model_name:
            raise ValueError(
                "No model specified. Please set model_name (for all methods) "
                "or base_model_name/grpo_model_name (for separate models)."
            )
        if needs_base_model:
            base_model_name = legacy_model_name
        if needs_grpo_model:
            grpo_model_name = legacy_model_name

    return (base_model_name if needs_base_model else None,
            grpo_model_name if needs_grpo_model else None)


def load_model_safely(
        model_name: str,
        device: Optional[str],
        dtype: Optional[str],
        model_type: str = "model",
) -> ModelWrapper:
    """
    Load a model with proper error handling for CUDA OOM.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load on (None for auto)
        dtype: Torch dtype string (e.g., "bfloat16", "float16")
        model_type: Description for error messages ("base model" or "GRPO model")
        
    Returns:
        Loaded ModelWrapper
    """
    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)

    try:
        print(f"Loading {model_type}: {model_name}")
        model_wrapper = ModelWrapper.from_pretrained(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        print(f"  Loaded successfully on device: {model_wrapper.device}")
        return model_wrapper

    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA out of memory when loading {model_type} '{model_name}'.\n\n"
            f"Original error: {e}"
        ) from e

    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} '{model_name}': {e}") from e


def unload_model(model_wrapper: Optional[ModelWrapper], name: str = "model"):
    """
    Unload a model and free GPU memory.
    
    Args:
        model_wrapper: ModelWrapper to unload
        name: Description for logging
    """
    if model_wrapper is not None:
        print(f"Unloading {name} to free memory...")
        del model_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def load_dataset(path: str, n_questions: Optional[int] = None) -> List[Dict]:
    """
    Load math questions from a JSON or JSONL file.
    
    Expected format: list of dicts with 'id', 'question', 'answer' fields.
    
    Args:
        path: Path to dataset file
        n_questions: Maximum number of questions to load
        
    Returns:
        List of question dicts
    """
    path = Path(path)

    if path.suffix == ".jsonl":
        data = load_jsonl(path)
    elif path.suffix == ".json":
        data = load_json(path)
        # Handle both list and dict formats
        if isinstance(data, dict):
            if "questions" in data:
                data = data["questions"]
            elif "data" in data:
                data = data["data"]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Ensure required fields
    for i, item in enumerate(data):
        if "id" not in item:
            item["id"] = str(i)
        if "question" not in item and "problem" in item:
            item["question"] = item["problem"]
        if "answer" not in item and "solution" in item:
            # Try to extract answer from solution
            item["answer"] = item.get("final_answer", item.get("solution", ""))

    if n_questions is not None:
        data = data[:n_questions]

    return data


def format_prompt(question: str, prompt_template: Optional[str] = None) -> str:
    """
    Format a math question into a prompt.
    
    Args:
        question: The math question text
        prompt_template: Optional template with {question} placeholder
        
    Returns:
        Formatted prompt string
    """
    if prompt_template:
        return prompt_template.format(question=question)

    # Default math prompt
    return f"""Solve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {question}

Solution:"""


def run_single_question(
        question_data: Dict,
        model_wrapper: ModelWrapper,
        method: str,
        config: Dict[str, Any],
        seed: int,
) -> Dict[str, Any]:
    """
    Run a method on a single question.
    
    Args:
        question_data: Dict with 'id', 'question', 'answer'
        model_wrapper: ModelWrapper instance
        method: Method name ('power_mcmc', 'grpo_single', 'grpo_vote')
        config: Configuration dict
        seed: Random seed
        
    Returns:
        Result dict with predictions and metrics
    """
    prompt = format_prompt(
        question_data["question"],
        config.get("prompt_template"),
    )

    gold_answer = question_data["answer"]

    if method == "power_mcmc":
        result = power_sampling_mcmc(
            prompt=prompt,
            model_wrapper=model_wrapper,
            alpha=config.get("alpha", 4.0),
            B=config.get("B", 192),
            N_mcmc=config.get("N_mcmc", 10),
            T_max=config.get("T_max", 1024),
            top_p=config.get("top_p", 1.0),
            seed=seed,
            verbose=config.get("verbose", False),
        )
    elif method == "grpo_single" or method == "grpo":
        result = run_grpo_single(
            prompt=prompt,
            model_wrapper=model_wrapper,
            max_new_tokens=config.get("T_max", 1024),
            temperature=config.get("baseline_temperature", 1.0),
            do_sample=config.get("do_sample", True),
            top_p=config.get("top_p", 1.0),
            seed=seed,
        )
    elif method == "grpo_vote" or method == "grpo_majority_vote":
        result = run_grpo_majority_vote(
            prompt=prompt,
            model_wrapper=model_wrapper,
            n_samples=config.get("n_vote_samples", 10),
            max_new_tokens=config.get("T_max", 1024),
            temperature=config.get("baseline_temperature", 1.0),
            do_sample=config.get("do_sample", True),
            top_p=config.get("top_p", 1.0),
            seed=seed,
            tie_break=config.get("tie_break", "logprob"),
            verbose=config.get("verbose", False),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check correctness
    is_correct = exact_match(result["final_answer"], gold_answer)

    # Build output row
    row = {
        "id": question_data["id"],
        "method": method,
        "is_correct": int(is_correct),
        "pred_answer": result["final_answer"],
        "gold_answer": gold_answer,
        "wall_time_s": result["wall_time_s"],
        "output_tokens": result["output_tokens"],
        "internal_generated_tokens": result.get("internal_generated_tokens", result["output_tokens"]),
        "accept_rate": result.get("accept_rate", None),
        "num_steps": result.get("num_steps", 0),
        "seed": seed,
        # Hyperparameters
        "alpha": config.get("alpha", 4.0) if method == "power_mcmc" else None,
        "B": config.get("B", 192) if method == "power_mcmc" else None,
        "N_mcmc": config.get("N_mcmc", 10) if method == "power_mcmc" else None,
        "T_max": config.get("T_max", 1024),
    }

    # Add raw outputs for debugging
    if config.get("save_outputs", False):
        row["raw_answer"] = result.get("raw_answer", "")
        row["output_text"] = result.get("output_text", "")[:500]  # Truncate

    return row


def run_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run a full experiment based on configuration.
    
    Supports both single-model and dual-model modes:
    - Single model: `model_name` used for all methods
    - Dual model: `base_model_name` for power_mcmc, `grpo_model_name` for baselines
    
    Args:
        config: Experiment configuration dict
        
    Returns:
        List of result rows
    """
    base_seed = config.get("seed", 42)
    set_seed(base_seed)

    # Get methods to run
    methods = config.get("methods", ["power_mcmc"])
    if isinstance(methods, str):
        methods = [methods]

    # Determine which models to load
    base_model_name, grpo_model_name = get_model_names_from_config(config, methods)

    # Check if we need both models (memory consideration)
    needs_both = base_model_name is not None and grpo_model_name is not None
    same_model = needs_both and base_model_name == grpo_model_name

    if needs_both and not same_model:
        print(f"\n{'=' * 60}")
        print("DUAL MODEL MODE")
        print(f"  Base model (power_mcmc): {base_model_name}")
        print(f"  GRPO model (baselines):  {grpo_model_name}")
        print(f"{'=' * 60}\n")

    # Load dataset
    dataset_path = config.get("dataset_path", "data/sample_math.jsonl")
    n_questions = config.get("n_questions", 50)
    questions = load_dataset(dataset_path, n_questions)
    print(f"Loading dataset: {dataset_path}, {len(questions)} questions")

    # For NMCMC sweep, we need to run power_mcmc multiple times
    nmcmc_values = config.get("nmcmc_sweep", None)

    results = []

    # Separate methods by model type
    power_mcmc_methods = [m for m in methods if m == "power_mcmc"]
    grpo_methods = [m for m in methods if m in ["grpo_single", "grpo", "grpo_vote", "grpo_majority_vote"]]

    # Track loaded models
    base_model_wrapper = None
    grpo_model_wrapper = None

    try:
        # --- Run power_mcmc with base model ---
        if power_mcmc_methods and base_model_name:
            base_model_wrapper = load_model_safely(
                base_model_name,
                config.get("device"),
                config.get("dtype"),
                "base model (for power_mcmc)",
            )

            for method in power_mcmc_methods:
                print(f"\n{'=' * 50}")
                print(f"Running method: {method}")
                print(f"Model: {base_model_name}")
                print(f"{'=' * 50}")

                if nmcmc_values:
                    # Sweep over NMCMC values
                    for nmcmc in nmcmc_values:
                        print(f"\n--- N_mcmc = {nmcmc} ---")
                        method_config = {**config, "N_mcmc": nmcmc}

                        for i, q in enumerate(questions):
                            print(f"  Question {i + 1}/{len(questions)}: {q['id']}")
                            seed = base_seed + i

                            try:
                                row = run_single_question(
                                    q, base_model_wrapper, method, method_config, seed
                                )
                                row["model_name"] = base_model_name
                                results.append(row)
                                print(f"    Correct: {row['is_correct']}, Time: {row['wall_time_s']:.2f}s")
                            except Exception as e:
                                print(f"    ERROR: {e}")
                                results.append({
                                    "id": q["id"],
                                    "method": method,
                                    "is_correct": 0,
                                    "pred_answer": "",
                                    "gold_answer": q["answer"],
                                    "error": str(e),
                                    "N_mcmc": nmcmc,
                                    "model_name": base_model_name,
                                })
                else:
                    # Single run without sweep
                    for i, q in enumerate(questions):
                        print(f"  Question {i + 1}/{len(questions)}: {q['id']}")
                        seed = base_seed + i

                        try:
                            row = run_single_question(q, base_model_wrapper, method, config, seed)
                            row["model_name"] = base_model_name
                            results.append(row)
                            print(f"    Correct: {row['is_correct']}, Time: {row['wall_time_s']:.2f}s")
                        except Exception as e:
                            print(f"    ERROR: {e}")
                            results.append({
                                "id": q["id"],
                                "method": method,
                                "is_correct": 0,
                                "pred_answer": "",
                                "gold_answer": q["answer"],
                                "error": str(e),
                                "model_name": base_model_name,
                            })

            # Free base model if we need to load GRPO model separately
            if grpo_methods and grpo_model_name and not same_model:
                unload_model(base_model_wrapper, "base model")
                base_model_wrapper = None

        # --- Run GRPO methods with GRPO model ---
        if grpo_methods and grpo_model_name:
            # Reuse base model if same, otherwise load GRPO model
            if same_model and base_model_wrapper is not None:
                grpo_model_wrapper = base_model_wrapper
                print(f"\nReusing same model for GRPO methods: {grpo_model_name}")
            else:
                grpo_model_wrapper = load_model_safely(
                    grpo_model_name,
                    config.get("device"),
                    config.get("dtype"),
                    "GRPO model (for baselines)",
                )

            for method in grpo_methods:
                print(f"\n{'=' * 50}")
                print(f"Running method: {method}")
                print(f"Model: {grpo_model_name}")
                print(f"{'=' * 50}")

                for i, q in enumerate(questions):
                    print(f"  Question {i + 1}/{len(questions)}: {q['id']}")
                    seed = base_seed + i

                    try:
                        row = run_single_question(q, grpo_model_wrapper, method, config, seed)
                        row["model_name"] = grpo_model_name
                        results.append(row)
                        print(f"    Correct: {row['is_correct']}, Time: {row['wall_time_s']:.2f}s")
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.append({
                            "id": q["id"],
                            "method": method,
                            "is_correct": 0,
                            "pred_answer": "",
                            "gold_answer": q["answer"],
                            "error": str(e),
                            "model_name": grpo_model_name,
                        })

    finally:
        # Clean up models
        if base_model_wrapper is not None:
            unload_model(base_model_wrapper, "base model")
        if grpo_model_wrapper is not None and grpo_model_wrapper is not base_model_wrapper:
            unload_model(grpo_model_wrapper, "GRPO model")

    return results


def main(config):
    # Run experiment
    results = run_experiment(config)

    # Save wilcoxon_test_mcmc_vs_standard
    timestamp = get_timestamp()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ensure_dir(args.output_dir / f"wilcoxon_test_mcmc_vs_standard_{timestamp}.csv")

    save_csv(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("=" * 50)

    for method in set(r["method"] for r in results):
        method_results = [r for r in results if r["method"] == method]
        accuracy = sum(r["is_correct"] for r in method_results) / len(method_results)
        avg_time = sum(r["wall_time_s"] for r in method_results if r.get("wall_time_s")) / len(method_results)

        # Get model name for this method
        model_names = set(r.get("model_name", "unknown") for r in method_results)
        model_str = ", ".join(model_names)

        print(f"\n{method}:")
        print(f"  Model: {model_str}")
        print(f"  Accuracy: {accuracy:.2%} ({sum(r['is_correct'] for r in method_results)}/{len(method_results)})")
        print(f"  Avg time: {avg_time:.2f}s")

        # MCMC-specific
        if method == "power_mcmc":
            nmcmc_groups = {}
            for r in method_results:
                n = r.get("N_mcmc", "?")
                if n not in nmcmc_groups:
                    nmcmc_groups[n] = []
                nmcmc_groups[n].append(r)

            if len(nmcmc_groups) > 1:
                print("\n  By N_mcmc:")
                for nmcmc in sorted(nmcmc_groups.keys()):
                    group = nmcmc_groups[nmcmc]
                    acc = sum(r["is_correct"] for r in group) / len(group)
                    time_avg = sum(r["wall_time_s"] for r in group if r.get("wall_time_s")) / len(group)
                    accept_rates = [r["accept_rate"] for r in group if r.get("accept_rate") is not None]
                    avg_accept = sum(accept_rates) / len(accept_rates) if accept_rates else 0
                    print(f"    N_mcmc={nmcmc}: acc={acc:.2%}, time={time_avg:.2f}s, accept={avg_accept:.2%}")


def load_config(path: str | None) -> dict:
    if not path:
        return {}

    config_path = Path(path)
    if config_path.suffix in [".yaml", ".yml"]:
        return load_yaml(config_path)
    return load_json(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Power Sampling experiments")
    parser.add_argument("--config", type=str, help="Path to config YAML/JSON file")
    parser.add_argument("--method", type=str, choices=["power_mcmc", "grpo_single", "grpo_vote"],
                        help="Method to run (overrides config)")
    parser.add_argument("--model", type=str, help="Model name/path for all methods (legacy mode)")
    parser.add_argument("--base_model", type=str, help="Base model for power_mcmc")
    parser.add_argument("--grpo_model", type=str, help="GRPO model for baselines")
    parser.add_argument("--dataset", type=str, help="Dataset path (overrides config)")
    parser.add_argument("--n_questions", type=int, help="Number of questions (overrides config)")
    parser.add_argument("--nmcmc", type=int, help="N_mcmc value (overrides config)")
    parser.add_argument("--alpha", type=float, help="Alpha value (overrides config)")
    parser.add_argument("--B", type=int, help="Block size (overrides config)")
    parser.add_argument("--T_max", type=int, help="Max sequence length (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (overrides config)")

    args = parser.parse_args()

    config = load_config(args.config)

    overrides = {
        "methods": [args.method] if args.method else None,
        "model_name": args.model,
        "base_model_name": args.base_model,
        "grpo_model_name": args.grpo_model,
        "dataset_path": args.dataset,
        "n_questions": args.n_questions,
        "N_mcmc": args.nmcmc,
        "alpha": args.alpha,
        "B": args.B,
        "T_max": args.T_max,
        "seed": args.seed,
        "dtype": args.dtype,
        "verbose": True if args.verbose else False
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    # Set defaults
    config.setdefault("methods", ["power_mcmc"])
    config.setdefault("alpha", 4.0)
    config.setdefault("B", 192)
    config.setdefault("N_mcmc", 10)
    config.setdefault("T_max", 1024)
    config.setdefault("top_p", 1.0)
    config.setdefault("n_questions", 50)
    config.setdefault("n_vote_samples", 10)
    config.setdefault("tie_break", "logprob")

    print("Configuration:")
    print(config)

    main(config)
