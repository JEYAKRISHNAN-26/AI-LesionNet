"""NIfTI grid-resampling, intensity harmonization, and native-space recovery.

WARNING: affine-header resampling aligns voxel grids but is not an independent
intensity-based image registration. Inputs should already be co-registered and
skull stripped, as in the manuscript's data preparation assumptions.
"""
import numpy as np
from scipy.ndimage import zoom


def _nifti_modules():
    import nibabel as nib
    from nibabel.processing import resample_from_to, resample_to_output
    return nib, resample_from_to, resample_to_output


def equalize_nonzero(x, bins=256):
    """Histogram CDF equalization applied only inside nonzero/brain voxels."""
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x) & (x != 0)
    if not np.any(mask):
        return np.zeros_like(x)
    vals = x[mask]
    lo, hi = np.percentile(vals, (0.5, 99.5))
    if hi <= lo:
        return x
    v = np.clip(vals, lo, hi)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= max(1, cdf[-1])
    out = np.zeros_like(x)
    out[mask] = np.interp(v, edges[1:], cdf).astype(np.float32)
    return out


def normalize(x, equalize=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, 0.)
    if equalize:
        x = equalize_nonzero(x)
    mask = x != 0
    if np.any(mask):
        mean = x[mask].mean()
        std = x[mask].std()
        x[mask] = (x[mask] - mean) / max(std, 1e-6)
    return x


def crop_pad(x, shape):
    """Symmetric crop/pad trailing 3 dimensions; return invertible mapping."""
    shape = tuple(int(v) for v in shape)
    original = x.shape[-3:]
    if len(shape) != 3:
        raise ValueError('shape must be [D,H,W]')
    source, target = [], []
    for n, t in zip(original, shape):
        length = min(n, t)
        src0 = max(0, (n - t) // 2)
        dst0 = max(0, (t - n) // 2)
        source.append(slice(src0, src0 + length))
        target.append(slice(dst0, dst0 + length))
    dest = np.zeros((*x.shape[:-3], *shape), dtype=x.dtype)
    dest[(..., *target)] = x[(..., *source)]
    mapping = {'native_dhw_shape': list(original),
               'src_bounds': [[s.start, s.stop] for s in source],
               'dst_bounds': [[s.start, s.stop] for s in target]}
    return dest, mapping


def _dhw(nifti):
    return np.asarray(nifti.dataobj, dtype=np.float32).transpose(2, 0, 1)


def load_case(t1_path, t2_path, flair_path, mask_path=None,
              shape=(32, 256, 256), spacing_mm=1., equalize=True):
    """Return processed [3,D,H,W], mask [D,H,W] or None, and inverse metadata."""
    nib, resample_from_to, resample_to_output = _nifti_modules()
    original_t1 = nib.load(str(t1_path))
    t1 = resample_to_output(nib.as_closest_canonical(original_t1), voxel_sizes=(spacing_mm,) * 3, order=1)
    reference = (t1.shape, t1.affine)
    volumes = [t1]
    for path in (t2_path, flair_path):
        other = nib.as_closest_canonical(nib.load(str(path)))
        volumes.append(resample_from_to(other, reference, order=1))
    normalized = np.stack([normalize(_dhw(v), equalize) for v in volumes])
    images, mapping = crop_pad(normalized, shape)
    target = None
    if mask_path is not None:
        mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
        mask_aligned = resample_from_to(mask_img, reference, order=0)
        target, _ = crop_pad((_dhw(mask_aligned) > 0.5).astype(np.uint8), shape)
    meta = dict(mapping, affine=t1.affine.tolist(),
                original_shape=list(original_t1.shape[:3]),
                original_affine=original_t1.affine.tolist(),
                spacing_mm=float(spacing_mm))
    return images.astype(np.float32), target, meta


def restore_prediction(prediction_dhw, meta, order=0):
    """Undo crop/padding and resample from isotropic T1 grid to original grid."""
    nib, resample_from_to, _ = _nifti_modules()
    cropped = np.asarray(prediction_dhw)
    restored = np.zeros(meta['native_dhw_shape'], dtype=np.float32)
    src = tuple(slice(*p) for p in meta['src_bounds'])
    dst = tuple(slice(*p) for p in meta['dst_bounds'])
    restored[src] = cropped[dst]
    img = nib.Nifti1Image(restored.transpose(1, 2, 0), np.asarray(meta['affine']))
    native = resample_from_to(img, (tuple(meta['original_shape']), np.asarray(meta['original_affine'])), order=order)
    array = np.asarray(native.dataobj)
    if order == 0:
        array = (array > 0.5).astype(np.uint8)
    return nib.Nifti1Image(array, np.asarray(meta['original_affine']))
