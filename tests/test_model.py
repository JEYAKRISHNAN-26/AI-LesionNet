import numpy as np
import torch
from ailesionnet.model import AILesionNet, predict_mc
from ailesionnet.losses import segmentation_loss
from ailesionnet.metrics import score_case
from ailesionnet.refinement import refine
from ailesionnet.preprocessing import crop_pad


def tiny_model():
    return AILesionNet(base_channels=4, embed_dim=32, heads=4,
                       layers=1, patch=4, dropout=0.3)


def test_forward_train_backward():
    torch.manual_seed(42)
    model = tiny_model()
    x = torch.randn(1, 3, 32, 32, 32)
    y = torch.zeros(1, 1, 32, 32, 32)
    y[:, :, 10:13, 15:20, 13:16] = 1
    prediction = model(x)
    assert prediction.shape == y.shape
    loss = segmentation_loss(prediction, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_stochastic_inference():
    model = tiny_model()
    x = torch.randn(1, 3, 32, 32, 32)
    mean, variance = predict_mc(model, x, iterations=3)
    assert mean.shape == variance.shape == (1, 1, 32, 32, 32)
    assert (variance >= 0).all()
    assert variance.max().item() > 0


def test_refinement_preserves_stable_connected_component():
    p = np.zeros((20, 20, 20), dtype=np.float32)
    p[7:12, 7:12, 7:12] = 0.95
    v = np.zeros_like(p)
    mask, weighted, unc = refine(p, v, min_component_voxels=3, closing_radius=0)
    assert mask.sum() == 125
    assert np.array_equal(weighted, p)
    assert not unc.any()


def test_metrics_perfect_prediction():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[3:6, 4:7, 4:7] = 1
    scores = score_case(mask, mask)
    assert scores['dice'] == scores['lesion_f1'] == 1.0
    assert scores['hd95_mm'] == 0.0


def test_crop_pad_mapping():
    x = np.ones((3, 7, 17, 9), dtype=np.float32)
    cropped, mapping = crop_pad(x, (8, 11, 12))
    assert cropped.shape == (3, 8, 11, 12)
    assert mapping['native_dhw_shape'] == [7, 17, 9]
