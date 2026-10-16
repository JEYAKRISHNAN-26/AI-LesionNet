"""NPZ dataset created by preprocess.py. Separate subjects across splits."""
import csv
import random
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    def __init__(self, manifest, split, augment=False):
        with open(manifest, newline='', encoding='utf8') as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} cases in {manifest}')
        self.augment = augment
        self.base = Path(manifest).resolve().parent

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['npz'])
        if not path.is_absolute():
            path = self.base / path
        with np.load(path) as f:
            x = f['image'].astype(np.float32).copy()  # [3,D,H,W]
            y = f['mask'].astype(np.float32).copy()  # [D,H,W]
        if self.augment:
            for dim in range(3):
                if random.random() < 0.5:
                    x = np.flip(x, axis=dim + 1).copy()
                    y = np.flip(y, axis=dim).copy()
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                # Rotate in-plane only: maintains shared image-mask transform.
                x = ndi.rotate(x, angle, axes=(2, 3), reshape=False, order=1, mode='nearest')
                y = ndi.rotate(y, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')
            if random.random() < 0.5:
                x = np.sign(x) * np.abs(x) ** random.uniform(0.9, 1.1)
            if random.random() < 0.5:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y[None]))
