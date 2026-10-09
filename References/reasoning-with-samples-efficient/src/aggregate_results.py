#!/usr/bin/env python3
"""
Aggregate benchmark results from multiple log files.
Parses tables and creates a unified comparison.
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """Extract benchmark results from a log file."""
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find the results table section
    # Look for lines like: | GSM8K | model | strategy | 60.0% | ...
    table_pattern = r'\| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \|'
    
    for match in re.finditer(table_pattern, content):
        benchmark = match.group(1).strip()
        model = match.group(2).strip()
        strategy = match.group(3).strip()
        pass_rate = match.group(4).strip()
        avg_time = match.group(5).strip()
        total_tokens = match.group(6).strip()
        avg_tokens = match.group(7).strip()
        total_cost = match.group(8).strip()
        cost_per_problem = match.group(9).strip()
        problems = match.group(10).strip()
        
        # Skip header rows
        if benchmark == 'Benchmark' or benchmark.startswith('==='):
            continue
        
        results.append({
            'log_file': Path(log_path).name,
            'benchmark': benchmark,
            'model': model,
            'strategy': strategy,
            'pass_rate': pass_rate,
            'avg_time': avg_time,
            'total_tokens': total_tokens,
            'avg_tokens': avg_tokens,
            'total_cost': total_cost,
            'cost_per_problem': cost_per_problem,
            'problems': problems
        })
    
    return results


def aggregate_results(log_dir: str = 'eval_logs') -> List[Dict[str, Any]]:
    """Aggregate results from all log files in directory."""
    all_results = []
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} not found")
        return []
    
    log_files = sorted(Path(log_dir).glob('*.log'))
    
    if not log_files:
        print(f"Error: No log files found in {log_dir}")
        return []
    
    for log_file in log_files:
        results = parse_log_file(str(log_file))
        all_results.extend(results)
    
    return all_results


def display_results(results: List[Dict[str, Any]]):
    """Display aggregated results in a nice table."""
    if not results:
        print("No results to display")
        return
    
    # Prepare data for tabulate
    headers = [
        'Experiment',
        'Strategy',
        'Pass Rate',
        'Avg Time (s)',
        'Total Tokens',
        'Avg Tokens',
        'Total Cost',
        'Cost/Problem',
        'Problems'
    ]
    
    rows = []
    for r in results:
        rows.append([
            r['log_file'].replace('.log', '').replace('exp_', ''),
            r['strategy'],
            r['pass_rate'],
            r['avg_time'],
            r['total_tokens'],
            r['avg_tokens'],
            r['total_cost'],
            r['cost_per_problem'],
            r['problems']
        ])
    
    print("=" * 150)
    print("AGGREGATED BENCHMARK RESULTS FROM ALL EXPERIMENTS")
    print("=" * 150)
    print()
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print()


def find_best_strategy(results: List[Dict[str, Any]]):
    """Find and highlight the best performing strategies."""
    if not results:
        return
    
    # Parse pass rates
    beam_results = []
    greedy_results = []
    
    for r in results:
        if 'BeamSearch' in r['strategy']:
            beam_results.append(r)
        elif 'Greedy' in r['strategy']:
            greedy_results.append(r)
    
    print("=" * 150)
    print("SUMMARY")
    print("=" * 150)
    print()
    
    if beam_results:
        # Find best beam search
        best_beam = max(beam_results, key=lambda x: float(x['pass_rate'].rstrip('%')))
        print(f"🏆 Best Beam Search Configuration:")
        print(f"   Strategy:   {best_beam['strategy']}")
        print(f"   Pass Rate:  {best_beam['pass_rate']}")
        print(f"   Avg Time:   {best_beam['avg_time']}")
        print(f"   Total Cost: {best_beam['total_cost']}")
        print()
    
    if greedy_results:
        # Average greedy performance
        avg_pass_rate = sum(float(r['pass_rate'].rstrip('%')) for r in greedy_results) / len(greedy_results)
        avg_time = sum(float(r['avg_time']) for r in greedy_results) / len(greedy_results)
        print(f"📊 Greedy Baseline (averaged across runs):")
        print(f"   Pass Rate:  {avg_pass_rate:.1f}%")
        print(f"   Avg Time:   {avg_time:.2f}s")
        print()
    
    if beam_results and greedy_results:
        best_beam_rate = float(best_beam['pass_rate'].rstrip('%'))
        improvement = best_beam_rate - avg_pass_rate
        print(f"📈 Improvement over Greedy: {improvement:+.1f}%")
        print()


def main():
    """Main entry point."""
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'eval_logs'
    
    print("Aggregating results...")
    print()
    
    results = aggregate_results(log_dir)
    
    if not results:
        print("No results found. Run experiments first:")
        print("  ./beam_search_evals.sh")
        return 1
    
    display_results(results)
    find_best_strategy(results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
