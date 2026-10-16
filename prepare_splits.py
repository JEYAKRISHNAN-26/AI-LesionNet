"""Generate subject-disjoint 70/15/15 split from a raw manifest."""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV subject_id,t1,t2,flair,mask')
    parser.add_argument('--output', default='data/cases_split.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with open(args.input, newline='', encoding='utf8') as file:
        records = list(csv.DictReader(file))
    required = {'subject_id', 't1', 't2', 'flair', 'mask'}
    if not records or not required.issubset(records[0]):
        raise ValueError(f'Expected columns {sorted(required)} and at least one case')
    subjects = sorted(set(r['subject_id'] for r in records))
    random.Random(args.seed).shuffle(subjects)
    n = len(subjects)
    train_cut, val_cut = int(n * 0.70), int(n * 0.85)
    assignment = {subject: ('train' if i < train_cut else 'val' if i < val_cut else 'test')
                  for i, subject in enumerate(subjects)}
    for row in records:
        row['split'] = assignment[row['subject_id']]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote {len(records)} scans from {n} subjects to {output}')


if __name__ == '__main__':
    main()
