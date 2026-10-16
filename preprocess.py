"""Convert NIfTI scans to cached, training-ready NPZ volumes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ailesionnet.preprocessing import load_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='CSV columns subject_id,t1,t2,flair,mask,split')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--shape', nargs=3, type=int, default=[32, 256, 256], metavar=('D','H','W'))
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--no-equalize', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline='', encoding='utf8') as file:
        rows = list(csv.DictReader(file))
    base = Path(args.manifest).resolve().parent
    results = []
    for n, row in enumerate(rows):
        def resolve(p):
            item = Path(p)
            return item if item.is_absolute() else (base / item)
        images, mask, meta = load_case(
            resolve(row['t1']), resolve(row['t2']), resolve(row['flair']),
            resolve(row['mask']), shape=args.shape, spacing_mm=args.spacing,
            equalize=not args.no_equalize,
        )
        name = f'{n:05d}_{row["subject_id"].replace("/", "_")}'
        np.savez_compressed(output / (name + '.npz'), image=images, mask=mask)
        (output / (name + '_meta.json')).write_text(json.dumps(meta, indent=2))
        results.append({'subject_id': row['subject_id'], 'split': row['split'],
                        'npz': name + '.npz'})
        print(f'[{n+1}/{len(rows)}] {row["subject_id"]} -> {name}.npz', flush=True)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'split', 'npz'])
        writer.writeheader()
        writer.writerows(results)
    print('Ready:', output / 'manifest.csv')


if __name__ == '__main__':
    main()
