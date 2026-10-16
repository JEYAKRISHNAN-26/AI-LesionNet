"""Voxel metrics, lesion-component F1 and physical-space HD95."""
import numpy as np
from scipy import ndimage as ndi


def score_case(prediction, target, voxel_spacing=(1., 1., 1.)):
    p = np.asarray(prediction) > 0
    g = np.asarray(target) > 0
    if p.shape != g.shape or p.ndim != 3:
        raise ValueError('Masks must have identical 3D shape')
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()
    dice = 1.0 if (p.sum() + g.sum()) == 0 else 2 * tp / max(1, p.sum() + g.sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else float(not g.any())
    recall = float(tp / (tp + fn)) if (tp + fn) else float(not p.any())
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    lp, npred = ndi.label(p)
    lg, ngold = ndi.label(g)
    matched_pairs = []
    for i in range(1, npred + 1):
        overlaps = lg[lp == i]
        ids, counts = np.unique(overlaps[overlaps > 0], return_counts=True)
        for j, count in zip(ids, counts):
            matched_pairs.append((int(count), i, int(j)))
    # Greedy one-to-one matching of overlapping lesion components.
    used_p, used_g = set(), set()
    for _, pi, gi in sorted(matched_pairs, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    matched = len(used_p)
    lesion_f1 = 1.0 if (npred + ngold) == 0 else 2 * matched / max(1, npred + ngold)
    if not p.any() and not g.any():
        hd95 = 0.0
    elif not p.any() or not g.any():
        hd95 = float('nan')  # Undefined; do not silently substitute a perfect 0.
    else:
        surface_p = p ^ ndi.binary_erosion(p)
        surface_g = g ^ ndi.binary_erosion(g)
        dist_to_g = ndi.distance_transform_edt(~surface_g, sampling=voxel_spacing)
        dist_to_p = ndi.distance_transform_edt(~surface_p, sampling=voxel_spacing)
        hd95 = float(np.percentile(np.concatenate([dist_to_g[surface_p], dist_to_p[surface_g]]), 95))
    return {'dice': float(dice), 'precision': precision, 'recall': recall,
            'lesion_f1': float(lesion_f1), 'false_positive_rate': fpr, 'hd95_mm': hd95}
