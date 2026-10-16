"""Decision-active uncertainty weighting, adaptive threshold, morphology."""
import numpy as np
from scipy import ndimage as ndi


def refine(probability, variance, alpha=0.7, beta=0.15, threshold=0.5,
           min_component_voxels=3, closing_radius=1):
    """Return binary mask, weighted probability, normalized uncertainty.

    alpha=0.7 is specified by the manuscript. The manuscript does not provide
    numeric beta/min-size/radius; these are documented tunable defaults.
    """
    probability = np.asarray(probability, dtype=np.float32)
    variance = np.asarray(variance, dtype=np.float32)
    if probability.shape != variance.shape or probability.ndim != 3:
        raise ValueError('Expected matching [D,H,W] maps')
    if not (0 <= threshold <= 1):
        raise ValueError('threshold outside [0,1]')
    # Variance is bounded by 0.25 for Bernoulli probabilities; normalize on
    # the known theoretical bound to avoid unstable sample-specific scaling.
    unc = np.clip(variance / 0.25, 0., 1.)
    weighted = probability * np.exp(-alpha * unc)
    mask = weighted >= (threshold + beta * unc)
    if closing_radius > 0:
        grid = np.ogrid[tuple(slice(-closing_radius, closing_radius + 1) for _ in range(3))]
        ball = sum(axis**2 for axis in grid) <= closing_radius**2
        # Apply small-radius closing; boundary voxels are otherwise preserved.
        mask = ndi.binary_closing(mask, structure=ball, border_value=0)
    if min_component_voxels > 1:
        labels, n = ndi.label(mask)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_component_voxels
        keep[0] = False
        mask = keep[labels] if n else np.zeros_like(mask, dtype=bool)
    return mask.astype(np.uint8), weighted, unc
