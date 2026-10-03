#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rws.utils import load_csv, ensure_dir


def load_results(input_path: str) -> List[Dict[str, Any]]:
    path = Path(input_path)
    
    if path.is_file():
        return load_csv(path)
    elif path.is_dir():
        all_results = []
        for csv_file in path.glob("*.csv"):
            all_results.extend(load_csv(csv_file))
        return all_results
    else:
        all_results = []
        for csv_file in glob.glob(input_path):
            all_results.extend(load_csv(csv_file))
        return all_results


def parse_numeric(value) -> Optional[float]:
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def plot_accuracy_vs_nmcmc(
    results: List[Dict],
    output_path: Path,
    title: str = "Accuracy vs N_mcmc (Power Sampling)",
) -> None:
    mcmc_results = [r for r in results if r.get("method") == "power_mcmc"]
    
    if not mcmc_results:
        print("No power_mcmc wilcoxon_test_mcmc_vs_standard found, skipping accuracy_vs_nmcmc plot")
        return
    
    nmcmc_groups = {}
    for r in mcmc_results:
        nmcmc = parse_numeric(r.get("N_mcmc"))
        if nmcmc is None:
            continue
        nmcmc = int(nmcmc)
        if nmcmc not in nmcmc_groups:
            nmcmc_groups[nmcmc] = []
        nmcmc_groups[nmcmc].append(r)
    
    if not nmcmc_groups:
        print("No N_mcmc values found, skipping plot")
        return
    
    nmcmc_values = sorted(nmcmc_groups.keys())
    accuracies = []
    stderrs = []
    
    for nmcmc in nmcmc_values:
        group = nmcmc_groups[nmcmc]
        correct = [int(r.get("is_correct", 0)) for r in group]
        acc = np.mean(correct)
        stderr = np.std(correct) / np.sqrt(len(correct)) if len(correct) > 1 else 0
        accuracies.append(acc)
        stderrs.append(stderr)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(nmcmc_values, accuracies, yerr=stderrs, 
                marker='o', markersize=8, capsize=5, capthick=2,
                linewidth=2, color='#2E86AB')
    
    ax.set_xlabel("N_mcmc (MH iterations per block)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xticks(nmcmc_values)
    ax.grid(True, alpha=0.3)

    for x, y in zip(nmcmc_values, accuracies):
        ax.annotate(f'{y:.1%}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_time_vs_accuracy(
    results: List[Dict],
    output_path: Path,
    title: str = "Time vs Accuracy Trade-off",
) -> None:
    groups = {}
    
    for r in results:
        method = r.get("method", "unknown")
        nmcmc = r.get("N_mcmc")
        
        if method == "power_mcmc" and nmcmc is not None:
            key = f"power_mcmc (N={int(parse_numeric(nmcmc))})"
        else:
            key = method
            
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    if not groups:
        print("No wilcoxon_test_mcmc_vs_standard to plot")
        return
    
    plot_data = []
    
    for key, group in groups.items():
        times = [parse_numeric(r.get("wall_time_s")) for r in group]
        times = [t for t in times if t is not None]
        correct = [int(r.get("is_correct", 0)) for r in group]
        
        if times and correct:
            avg_time = np.mean(times)
            accuracy = np.mean(correct)
            time_stderr = np.std(times) / np.sqrt(len(times)) if len(times) > 1 else 0
            acc_stderr = np.std(correct) / np.sqrt(len(correct)) if len(correct) > 1 else 0
            
            plot_data.append({
                "label": key,
                "time": avg_time,
                "accuracy": accuracy,
                "time_err": time_stderr,
                "acc_err": acc_stderr,
                "n_samples": len(group),
            })
    
    if not plot_data:
        print("No valid data to plot")
        return
    
    plot_data.sort(key=lambda x: x["time"])
    
    colors = {
        "grpo_single": "#E63946",
        "grpo_vote": "#F4A261",
        "grpo_majority_vote": "#F4A261",
    }
    
    mcmc_entries = [p for p in plot_data if "power_mcmc" in p["label"]]
    if mcmc_entries:
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(mcmc_entries)))
        for i, entry in enumerate(mcmc_entries):
            colors[entry["label"]] = blues[i]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for data in plot_data:
        color = colors.get(data["label"], "#666666")
        
        ax.errorbar(
            data["time"], data["accuracy"],
            xerr=data["time_err"], yerr=data["acc_err"],
            marker='o', markersize=10, capsize=5,
            label=data["label"], color=color,
            linewidth=0, elinewidth=2,
        )
    
    mcmc_data = [d for d in plot_data if "power_mcmc" in d["label"]]
    if len(mcmc_data) > 1:
        mcmc_data.sort(key=lambda x: x["time"])
        times = [d["time"] for d in mcmc_data]
        accs = [d["accuracy"] for d in mcmc_data]
        ax.plot(times, accs, '--', color='#2E86AB', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("Average Wall Time (seconds)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    for data in plot_data:
        ax.annotate(
            f'{data["accuracy"]:.1%}',
            (data["time"], data["accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            alpha=0.8,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_token_efficiency(
    results: List[Dict],
    output_path: Path,
    title: str = "Token Efficiency: Accuracy vs Total Tokens",
) -> None:
    groups = {}
    
    for r in results:
        method = r.get("method", "unknown")
        nmcmc = r.get("N_mcmc")
        
        if method == "power_mcmc" and nmcmc is not None:
            key = f"power_mcmc (N={int(parse_numeric(nmcmc))})"
        else:
            key = method
            
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    plot_data = []
    
    for key, group in groups.items():
        total_tokens = []
        for r in group:
            internal = parse_numeric(r.get("internal_generated_tokens"))
            output = parse_numeric(r.get("output_tokens"))
            if internal is not None:
                total_tokens.append(internal)
            elif output is not None:
                total_tokens.append(output)
                
        correct = [int(r.get("is_correct", 0)) for r in group]
        
        if total_tokens and correct:
            avg_tokens = np.mean(total_tokens)
            accuracy = np.mean(correct)
            
            plot_data.append({
                "label": key,
                "tokens": avg_tokens,
                "accuracy": accuracy,
            })
    
    if not plot_data:
        print("No valid data for token efficiency plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
    
    for i, data in enumerate(plot_data):
        ax.scatter(
            data["tokens"], data["accuracy"],
            s=150, label=data["label"], color=colors[i],
            edgecolors='black', linewidth=1,
        )
        ax.annotate(
            data["label"].replace("power_mcmc ", "").replace("(", "").replace(")", ""),
            (data["tokens"], data["accuracy"]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
        )
    
    ax.set_xlabel("Average Total Tokens Generated", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_acceptance_rate(
    results: List[Dict],
    output_path: Path,
    title: str = "MH Acceptance Rate vs N_mcmc",
) -> None:
    mcmc_results = [r for r in results 
                    if r.get("method") == "power_mcmc" 
                    and parse_numeric(r.get("accept_rate")) is not None]
    
    if not mcmc_results:
        print("No acceptance rate data, skipping plot")
        return
    
    nmcmc_groups = {}
    for r in mcmc_results:
        nmcmc = int(parse_numeric(r.get("N_mcmc", 0)))
        if nmcmc not in nmcmc_groups:
            nmcmc_groups[nmcmc] = []
        nmcmc_groups[nmcmc].append(parse_numeric(r.get("accept_rate")))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nmcmc_values = sorted(nmcmc_groups.keys())
    accept_rates = [np.mean(nmcmc_groups[n]) for n in nmcmc_values]
    stderrs = [np.std(nmcmc_groups[n]) / np.sqrt(len(nmcmc_groups[n])) 
               for n in nmcmc_values]
    
    ax.errorbar(nmcmc_values, accept_rates, yerr=stderrs,
                marker='s', markersize=8, capsize=5,
                linewidth=2, color='#E07A5F')
    
    ax.set_xlabel("N_mcmc", fontsize=12)
    ax.set_ylabel("Acceptance Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xticks(nmcmc_values)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment wilcoxon_test_mcmc_vs_standard")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input CSV file or directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for figures")
    parser.add_argument("--plots", type=str, nargs="+",
                        choices=["accuracy", "time", "tokens", "accept", "all"],
                        default=["all"],
                        help="Which plots to generate")
    
    args = parser.parse_args()
    print(f"Loading wilcoxon_test_mcmc_vs_standard from: {args.input}")
    results = load_results(args.input)
    print(f"Loaded {len(results)} result rows")
    
    if not results:
        print("No wilcoxon_test_mcmc_vs_standard found!")
        return
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "wilcoxon_test_mcmc_vs_standard" / "figures"
    ensure_dir(output_dir)
    
    plots_to_make = set(args.plots)
    if "all" in plots_to_make:
        plots_to_make = {"accuracy", "time", "tokens", "accept"}
    
    if "accuracy" in plots_to_make:
        plot_accuracy_vs_nmcmc(
            results,
            output_dir / "accuracy_vs_nmcmc.png"
        )
    
    if "time" in plots_to_make:
        plot_time_vs_accuracy(
            results,
            output_dir / "time_vs_accuracy.png"
        )
    
    if "tokens" in plots_to_make:
        plot_token_efficiency(
            results,
            output_dir / "token_efficiency.png"
        )
    
    if "accept" in plots_to_make:
        plot_acceptance_rate(
            results,
            output_dir / "acceptance_rate.png"
        )
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
