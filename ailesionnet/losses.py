"""Combined BCE and soft Dice loss for sparse lesion masks."""
import torch
from torch.nn import functional as F


def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    overlap = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * overlap + eps) / (denominator + eps)).mean()


def segmentation_loss(logits, targets, dice_weight=0.5):
    if not (0 <= dice_weight <= 1):
        raise ValueError('dice_weight must be in [0,1]')
    return (dice_weight * soft_dice_loss(logits, targets)
            + (1 - dice_weight) * F.binary_cross_entropy_with_logits(logits, targets))
