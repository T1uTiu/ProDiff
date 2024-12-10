import os
from typing import List
import matplotlib
import numpy as np


matplotlib.use('Agg')

import glob
import importlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
from tasks.base_task import BaseDataset
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class DurPredictorDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = os.path.join(hparams['data_dir'], hparams["task"]) 
        self.prefix = prefix
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def __getitem__(self, index) -> dict:
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        
        batch_item = {
            "nsamples" : len(samples),
            "ph_seq" : utils.collate_1d([torch.LongTensor(s["ph_seq"]) for s in samples], 0),
            "ph_dur": utils.collate_1d([torch.FloatTensor(s["ph_dur"]) for s in samples], 0.0),
            "word_dur": utils.collate_1d([torch.FloatTensor(s["word_dur"]) for s in samples], 0.0),
            "onset": utils.collate_1d([torch.LongTensor(s["onset"]) for s in samples], 0),
        }
        
        return batch_item