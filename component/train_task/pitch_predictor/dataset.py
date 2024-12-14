from typing import List
import torch
from component.train_task.base_dataset import BaseDataset
import utils

class PitchPredictorDataset(BaseDataset):
    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        
        batch_item = {
            "nsamples" : len(samples),
            "note_midi": utils.collate_1d([torch.Tensor(s["note_midi"]) for s in samples], -1),
            "note_rest": utils.collate_1d([torch.BoolTensor(s["note_rest"]) for s in samples], True),
            "mel2note" : utils.collate_1d([torch.LongTensor(s["mel2note"]) for s in samples], 0),
            "pitch": utils.collate_1d([torch.FloatTensor(s["pitch"]) for s in samples], 0.0),
            "f0" : utils.collate_1d([torch.FloatTensor(s["f0"]) for s in samples], 0.0),
            "base_pitch" : utils.collate_1d([torch.FloatTensor(s["base_pitch"]) for s in samples], 0.0),
            "base_f0" : utils.collate_1d([torch.FloatTensor(s["base_f0"]) for s in samples], 0.0),
        }
        if self.hparams['use_spk_id']:
            batch_item["spk_id"] = torch.LongTensor([s["spk_id"] for s in samples])
        return batch_item