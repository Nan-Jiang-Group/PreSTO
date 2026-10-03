import argparse
import ast
import csv
import os
import pandas as pd


def parse_list_column(series):
    def parse_cell(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return series.apply(parse_cell)


def main():
    ap = argparse.ArgumentParser(description="Compare KCBS vs Best-of-N outputs and save side-by-side CSV")
    ap.add_argument("--bestof_csv", required=True)
    ap.add_argument("--kcbs_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k", type=int, default=2, help="Number of sequences (for labeling)")
    args = ap.parse_args()

    bo = pd.read_csv(args.bestof_csv)
    kc = pd.read_csv(args.kcbs_csv)

    # Parse list-like fields safely
    for df in (bo, kc):
        if 'completions' in df.columns:
            df['completions'] = parse_list_column(df['completions'])
        if 'answers' in df.columns:
            df['answers'] = parse_list_column(df['answers'])
        if 'correct_flags' in df.columns:
            df['correct_flags'] = parse_list_column(df['correct_flags'])

    # Inner join on question
    merged = pd.merge(bo, kc, on='question', suffixes=('_bestof', '_kcbs'), how='inner')

    rows = []
    for _, row in merged.iterrows():
        question = row['question']
        correct = row.get('correct_answer_bestof', row.get('correct_answer_kcbs', ''))

        bo_comps = row.get('completions_bestof', [])
        kc_comps = row.get('completions_kcbs', [])

        # Normalize to 2 slots
        bo_c1 = bo_comps[0] if len(bo_comps) > 0 else ''
        bo_c2 = bo_comps[1] if len(bo_comps) > 1 else ''
        kc_c1 = kc_comps[0] if len(kc_comps) > 0 else ''
        kc_c2 = kc_comps[1] if len(kc_comps) > 1 else ''

        bo_flags = row.get('correct_flags_bestof', [])
        kc_flags = row.get('correct_flags_kcbs', [])
        bo_flag_any = int(max(bo_flags)) if isinstance(bo_flags, list) and bo_flags else 0
        kc_flag_any = int(max(kc_flags)) if isinstance(kc_flags, list) and kc_flags else 0

        out_row = {
            'question': question,
            'correct_answer': correct,
            'bestof_text_1': bo_c1,
            'bestof_text_2': bo_c2,
            'kcbs_text_1': kc_c1,
            'kcbs_text_2': kc_c2,
            'bestof_any_correct': bo_flag_any,
            'kcbs_any_correct': kc_flag_any,
        }
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote comparison CSV to {args.out_csv} with {len(out_df)} rows")


if __name__ == '__main__':
    main()

