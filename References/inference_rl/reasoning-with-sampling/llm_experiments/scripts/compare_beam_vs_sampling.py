import argparse
import pandas as pd
import matplotlib.pyplot as plt


def compute_pass_curve(csv_path):
    df = pd.read_csv(csv_path)
    df['correct_flags'] = df['correct_flags'].apply(eval)
    k = df['k'].iloc[0]
    pass_at_j = []
    for j in range(1, k + 1):
        scores = []
        for flags in df['correct_flags']:
            scores.append(max(flags[:j]))
        pass_at_j.append((j, sum(scores) / len(scores)))
    return pass_at_j


def plot_curves(beam_curve, sampling_curve, output_path, title):
    plt.figure(figsize=(8, 6))
    beam_x, beam_y = zip(*beam_curve)
    sampling_x, sampling_y = zip(*sampling_curve)
    plt.plot(beam_x, beam_y, 'o-', label='Beam search', linewidth=2)
    plt.plot(sampling_x, sampling_y, 's-', label='Best-of-N sampling', linewidth=2)
    plt.xlabel('Best of N (k)')
    plt.ylabel('Pass@k Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(max(beam_x), max(sampling_x)) + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare beam search vs sampling pass@k curves')
    parser.add_argument('--beam_csv', required=True, help='Path to beam search CSV')
    parser.add_argument('--sampling_csv', required=True, help='Path to sampling CSV')
    parser.add_argument('--output', required=True, help='Output plot path')
    parser.add_argument('--title', default='Beam Search vs Best-of-N Sampling', help='Plot title')
    args = parser.parse_args()

    beam_curve = compute_pass_curve(args.beam_csv)
    sampling_curve = compute_pass_curve(args.sampling_csv)

    plot_curves(beam_curve, sampling_curve, args.output, args.title)
    print(f'Comparison plot saved to: {args.output}')


if __name__ == '__main__':
    main()
