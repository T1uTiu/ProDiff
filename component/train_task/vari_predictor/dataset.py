import os
from typing import List
import matplotlib

from component.binarizer.base import get_binarizer_cls
matplotlib.use('Agg')

import glob
import importlib
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
from tasks.base_task import BaseDataset
from utils.cwt import get_lf0_cwt
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from handler.binarize.handler import PreprocessedItem


class VariPredictorDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        binarizer_cls = get_binarizer_cls(hparams)
        self.data_dir = os.path.join(hparams['data_dir'], binarizer_cls.category()) 
        self.prefix = prefix
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

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'preprocess.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes
