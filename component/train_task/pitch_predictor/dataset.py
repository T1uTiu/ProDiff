import os
from typing import List
import numpy as np
import torch
from component.train_task.base_dataset import BaseDataset
import utils
from utils.indexed_datasets import IndexedDataset


class PitchPredictorDataset(BaseDataset):
    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        
        batch_item = {
            "nsamples" : len(samples),
            "note_midi": utils.collate_1d([torch.Tensor(s["note_midi"]) for s in samples], -1),
            "mel2note" : utils.collate_1d([torch.LongTensor(s["mel2note"]) for s in samples], 0),
            "f0" : utils.collate_1d([torch.FloatTensor(s["f0"]) for s in samples], 0.0),
            "base_pitch" : utils.collate_1d([torch.FloatTensor(s["base_pitch"]) for s in samples], 0.0),
            "base_f0" : utils.collate_1d([torch.FloatTensor(s["base_f0"]) for s in samples], 0.0),
        }
        if self.hparams['use_spk_id']:
            batch_item["spk_id"] = torch.LongTensor([s["spk_id"] for s in samples])
        
        return batch_item