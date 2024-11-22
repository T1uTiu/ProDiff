from typing import List
import matplotlib



matplotlib.use('Agg')

import glob
import importlib
import os
from dataclasses import dataclass
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
from utils.pitch_utils import norm_interp_f0
from preprocess.base_binarizer import TranscriptionItem

@dataclass
class FastSpeechDataSetItem:
    id: int
    item_name: str
    text: str

    phoeneme: torch.LongTensor
    mel2ph: torch.LongTensor

    speaker_id: int = -1
    lang_id: int = -1

    f0: torch.FloatTensor
    uv: torch.FloatTensor
    pitch: torch.LongTensor

    mel: torch.Tensor
    mel_nonpadding: torch.Tensor

class FastSpeechDataset(BaseDataset):
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

    def _get_item(self, index) -> TranscriptionItem:
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item.mel)[:max_frames]
        mel2ph = torch.LongTensor(item.mel2ph)[:max_frames]
        f0 = item.f0[:max_frames]
        uv = f0 == 0
        f0, uv = torch.FloatTensor(f0), torch.FloatTensor(uv)
        phone = torch.LongTensor(item.phoneme[:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.pitch)[:max_frames]
        sample = FastSpeechDataSetItem(
            id=index,
            item_name=item.item_name,
            text=item.text,
            phoeneme=phone,
            mel2ph=mel2ph,
            f0=f0,
            uv=uv,
            pitch=pitch,
            mel=spec,
            mel_nonpadding=spec.abs().sum(-1) > 0,
        )
        if self.hparams['use_spk_id']:
            sample.speaker_id = item.spk_id
        if self.hparams['use_lang_id']:
            sample.lang_id = item.lang_id
        return sample

    def collater(self, samples: List[FastSpeechDataSetItem]):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s.id for s in samples])
        item_names = [s.item_name for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s.phoeneme for s in samples], 0)
        f0 = utils.collate_1d([s.f0 for s in samples], 0.0)
        pitch = utils.collate_1d([s.pitch for s in samples])
        uv = utils.collate_1d([s.uv for s in samples])
        # energy = utils.collate_1d([s.energy for s in samples], 0.0)
        mel2ph = utils.collate_1d([s.mel2ph for s in samples], 0.0) \
            if samples[0].mel2ph is not None else None
        mels = utils.collate_2d([s.mel for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s.phoeneme.numel() for s in samples])
        mel_lengths = torch.LongTensor([s.mel.shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            # 'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }

        if self.hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s.speaker_id for s in samples])
            batch['spk_ids'] = spk_ids
        if self.hparams['use_lang_id']:
            lang_ids = torch.LongTensor([s.lang_id for s in samples])
            batch['lang_ids'] = lang_ids
        return batch

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
