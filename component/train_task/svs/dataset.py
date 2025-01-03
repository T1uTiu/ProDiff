import os
from typing import List
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
import utils
from component.train_task.base_dataset import BaseDataset


class SVSDataset(BaseDataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__(prefix, shuffle, hparams)
        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None



    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        
        batch_item = {
            "nsamples" : len(samples),
            "ph_seq" : utils.collate_1d([torch.LongTensor(s["ph_seq"]) for s in samples], 0),
            "mel2ph" : utils.collate_1d([torch.LongTensor(s["mel2ph"]) for s in samples], 0),
            "f0" : utils.collate_1d([torch.FloatTensor(s["f0"]) for s in samples], 0.0),
            "mel" : utils.collate_2d([torch.Tensor(s["mel"]) for s in samples], 0.0)
        }
        
        if self.hparams['use_spk_id']:
            batch_item["spk_id"] = torch.LongTensor([s["spk_id"] for s in samples])

        if self.hparams["use_gender_id"]:
            batch_item["gender_id"] = torch.LongTensor([s["gender_id"] for s in samples])
        
        if self.hparams['use_lang_id']:
            batch_item["lang_seq"] = utils.collate_1d([torch.LongTensor(s["lang_seq"]) for s in samples], 0)

        if self.hparams["use_voicing_embed"]:
            batch_item["voicing"] = utils.collate_1d([torch.FloatTensor(s["voicing"]) for s in samples], 0.0)
        
        if self.hparams["use_breath_embed"]:
            batch_item["breath"] = utils.collate_1d([torch.FloatTensor(s["breath"]) for s in samples], 0.0)
        
        return batch_item
