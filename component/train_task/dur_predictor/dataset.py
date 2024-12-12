import os
from typing import List
import matplotlib
import numpy as np

from component.train_task.base_dataset import BaseDataset


matplotlib.use('Agg')

import glob
import importlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class DurPredictorDataset(BaseDataset):
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