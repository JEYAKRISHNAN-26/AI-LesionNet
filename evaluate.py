"""Score ground-truth vs predicted NIfTI masks; no hardcoded paper results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ailesionnet.metrics import score_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='CSV: subject_id,mask,prediction (paths)')
    parser.add_argument('--output', default='evaluation/metrics.csv')
    args = parser.parse_args()
    import nibabel as nib
    with open(args.manifest, newline='', encoding='utf8') as f:
        cases = list(csv.DictReader(f))
    base = Path(args.manifest).resolve().parent
    rows = []
    for case in cases:
        def resolve(p):
            item = Path(p)
            return item if item.is_absolute() else base / item
        label_img = nib.load(str(resolve(case['mask'])))
        predicted_img = nib.load(str(resolve(case['prediction'])))
        if label_img.shape != predicted_img.shape or not np.allclose(label_img.affine, predicted_img.affine, atol=1e-3):
            raise ValueError(f'{case["subject_id"]}: prediction not aligned to ground truth grid')
        spacing = nib.affines.voxel_sizes(label_img.affine)[:3]
        truth = np.asarray(label_img.dataobj) > 0.5
        pred = np.asarray(predicted_img.dataobj) > 0.5
        metrics = score_case(pred, truth, voxel_spacing=spacing)
        rows.append({'subject_id': case['subject_id'], **metrics})
    if not rows:
        raise ValueError('No cases to evaluate')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = {k: {'mean': float(np.nanmean([r[k] for r in rows])),
                 'std': float(np.nanstd([r[k] for r in rows]))}
             for k in rows[0] if k != 'subject_id'}
    stats['number_of_cases'] = len(rows)
    (output.parent / 'summary.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
