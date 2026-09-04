import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=48, samples_per_epoch=200, fossil_bias=0.7, train=True):
        self.vol_paths = sorted(glob.glob(os.path.join(data_dir, "vol_*.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, "mask_*.npy")))
        assert len(self.vol_paths) == len(self.mask_paths) and len(self.vol_paths) > 0
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.fossil_bias = fossil_bias
        self.train = train
        self._cache = {}

    def __len__(self):
        return self.samples_per_epoch

    def _load(self, idx):
        if idx not in self._cache:
            vol = np.load(self.vol_paths[idx])
            mask = np.load(self.mask_paths[idx])
            self._cache[idx] = (vol, mask)
        return self._cache[idx]

    def _random_crop(self, vol, mask, ps):
        shape = vol.shape
        if self.train and np.random.rand() < self.fossil_bias and mask.sum() > 0:
            coords = np.argwhere(mask)
            c = coords[np.random.randint(len(coords))]
            lo = [max(0, min(c[d] - ps // 2 + np.random.randint(-ps // 4, ps // 4), shape[d] - ps)) for d in range(3)]
        else:
            lo = [np.random.randint(0, shape[d] - ps + 1) for d in range(3)]
        sl = tuple(slice(lo[d], lo[d] + ps) for d in range(3))
        return vol[sl], mask[sl]

    def _augment(self, vol, mask):
        if np.random.rand() < 0.5:
            vol = np.flip(vol, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() < 0.5:
            vol = np.flip(vol, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() < 0.5:
            vol = np.flip(vol, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        k = np.random.randint(4)
        if k:
            vol = np.rot90(vol, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        if np.random.rand() < 0.3:
            vol = vol + np.random.normal(scale=0.01, size=vol.shape).astype(np.float32)
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.85, 1.15)
            vol = np.clip(vol, 0, 1) ** gamma
        return vol.astype(np.float32), mask.astype(np.uint8)

    def __getitem__(self, i):
        idx = np.random.randint(len(self.vol_paths))
        vol, mask = self._load(idx)
        ps = self.patch_size
        vol_p, mask_p = self._random_crop(vol, mask, ps)
        if self.train:
            vol_p, mask_p = self._augment(vol_p, mask_p)
        vol_t = torch.from_numpy(vol_p).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_p.astype(np.int64))
        return vol_t, mask_t
