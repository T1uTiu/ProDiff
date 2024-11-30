from dataclasses import dataclass
import os
from typing import List
import matplotlib
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

@dataclass
class ProDiffDatasetBatchItem:
    nsamples: int

    spk_id: torch.LongTensor = None
    lang_seq: torch.LongTensor = None
    ph_seq: torch.LongTensor = None
    mel2ph: torch.LongTensor = None
    f0: torch.FloatTensor = None
    txt_lengths: torch.LongTensor = None
    mel_lengths: torch.LongTensor = None

    mel: torch.Tensor = None

    def to(self, device, non_blocking=False, copy=False):
        for attr_name in self.__dict__:
            attr = getattr(self, attr_name)
            if callable(getattr(attr, 'to', None)):
                setattr(self, attr_name, attr.to(device, non_blocking=non_blocking, copy=copy))
        return self

class ProDiffDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if hparams['pitch_type'] == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def __getitem__(self, index) -> PreprocessedItem:
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def collater(self, samples: List[PreprocessedItem]):
        if len(samples) == 0:
            return ProDiffDatasetBatchItem()
        
        batch_item = ProDiffDatasetBatchItem(
            nsamples = len(samples),
            ph_seq = utils.collate_1d([torch.LongTensor(s.ph_seq) for s in samples], 0),
            mel2ph = utils.collate_1d([torch.LongTensor(s.mel2ph) for s in samples], 0),
            f0 = utils.collate_1d([torch.FloatTensor(s.f0) for s in samples], 0.0),
            
            mel = utils.collate_2d([torch.Tensor(s.mel) for s in samples], 0.0)
        )

        batch_item.txt_lengths = torch.LongTensor([s.ph_seq.size for s in samples])
        batch_item.mel_lengths = torch.LongTensor([s.mel.shape[0] for s in samples])
        
        if self.hparams['use_spk_id']:
            batch_item.spk_id = torch.LongTensor([s.spk_id for s in samples])
        
        if self.hparams['use_lang_id']:
            batch_item.lang_seq = utils.collate_1d([torch.LongTensor(s.lang_seq) for s in samples], 0)
        
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
