import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def eval_vllm_passk_results(csv_file):
    """Evaluate pass@k results from vLLM batched inference"""
    df = pd.read_csv(csv_file)
    
    # Parse the stored lists from CSV (they're stored as strings)
    df['correct_flags'] = df['correct_flags'].apply(eval)
    df['answers'] = df['answers'].apply(eval)
    
    k = df['k'].iloc[0]
    total_problems = len(df)
    
    print(f"Evaluating Pass@{k} results:")
    print(f"Total problems: {total_problems}")
    
    # Overall pass@k accuracy (already computed during generation)
    overall_pass_at_k = df['pass_at_k'].mean()
    print(f"Pass@{k} Accuracy: {overall_pass_at_k:.4f}")
    
    # Compute pass@j for j = 1, 2, ..., k
    pass_at_j_results = []
    
    for j in range(1, k + 1):
        pass_at_j_scores = []
        
        for _, row in df.iterrows():
            correct_flags = row['correct_flags']
            # For pass@j, take the best result from the first j samples
            pass_at_j_score = max(correct_flags[:j])
            pass_at_j_scores.append(pass_at_j_score)
        
        pass_at_j_accuracy = np.mean(pass_at_j_scores)
        pass_at_j_results.append((j, pass_at_j_accuracy))
        print(f"Pass@{j}: {pass_at_j_accuracy:.4f}")
    
    return pass_at_j_results


def plot_passk_curve(pass_at_j_results, output_file=None):
    """Plot pass@k curve"""
    js = [result[0] for result in pass_at_j_results]
    accuracies = [result[1] for result in pass_at_j_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(js, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('k (number of samples)')
    plt.ylabel('Pass@k Accuracy')
    plt.title('Pass@k Performance')
    plt.grid(True, alpha=0.3)
    plt.xticks(js)
    
    # Add value labels on points
    for j, acc in zip(js, accuracies):
        plt.annotate(f'{acc:.3f}', (j, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Don't show plot interactively - just save to file
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Path to CSV file with vLLM pass@k results")
    parser.add_argument("--plot", action="store_true", help="Generate pass@k curve plot")
    parser.add_argument("--output_plot", type=str, help="Output file for plot")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found")
        return
    
    print(f"Evaluating results from: {args.csv_file}")
    pass_at_j_results = eval_vllm_passk_results(args.csv_file)
    
    if args.plot:
        output_plot = args.output_plot
        if not output_plot:
            # Generate default plot filename
            csv_path = Path(args.csv_file)
            output_plot = csv_path.parent / f"{csv_path.stem}_plot.png"
        
        plot_passk_curve(pass_at_j_results, output_plot)


if __name__ == "__main__":
    main()