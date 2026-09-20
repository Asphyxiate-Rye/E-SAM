"""BTCV/Synapse dataset for the E-SAM training loop.

Upstream ships no BTCV code (checked: no commit in this repo's history ever
added one), so this mirrors dataset.py's MMWHS_dataset -- same npz-per-slice
/ h5-per-volume layout, same RandomGenerator augmentation -- with the two
MMWHS-specific steps removed:

  - No HU rewindowing. MMWHS_dataset does (x+750)/1500 for a cardiac window;
    prepare_btcv.py/prepare_synapse_ct.py/prepare_acdc.py have already
    applied their own normalization ([-125,275]->[0,1] HU window for the
    CT datasets, percentile-clip for ACDC MRI), so the values arrive ready
    to use.
  - No label remap. MMWHS labels are 205/420/500/... and need mapping to
    1..7; BTCV labels are already integers 1..13.
"""
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from dataset import RandomGenerator  # noqa: F401  (re-exported for trainer)


class BTCV_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data = np.load(os.path.join(self.data_dir, slice_name + '.npz'))
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            data = h5py.File(self.data_dir + "/{}.h5".format(vol_name))
            image, label = data['image'][:], data['label'][:]
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
