from itertools import chain
import os
import torch
from torch.utils.data import ConcatDataset
import numpy as np

from utils.indexed_datasets import IndexedDataset

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.data_dir = os.path.join(hparams['data_dir'], hparams["task"]) 
        self.prefix = prefix
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], self.hparams['max_frames'])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))


class BaseConcatDataset(ConcatDataset):
    def __init__(self, datasets, hparams):
        super().__init__(datasets)
        self.hparams = hparams

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, 'sizes'):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], self.hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return self.datasets[0].num_workers