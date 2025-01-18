from typing import List
import torch
from component.train_task.base_dataset import BaseDataset
import utils
from utils.pitch_utils import random_continuous_masks

def random_retake_masks(b, t, device):
    # 1/4 segments are True in average
    B_masks = torch.randint(low=0, high=4, size=(b, 1), dtype=torch.long, device=device) == 0
    # 1/3 frames are True in average
    T_masks = random_continuous_masks(b, t, dim=1, device=device)
    # 1/4 segments and 1/2 frames are True in average (1/4 + 3/4 * 1/3 = 1/2)

class PitchPredictorDataset(BaseDataset):
    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        
        batch_item = {
            "nsamples" : len(samples),
            "ph_seq" : utils.collate_1d([torch.LongTensor(s["ph_seq"]) for s in samples], 0),
            "mel2ph" : utils.collate_1d([torch.LongTensor(s["mel2ph"]) for s in samples], 0),
            "note_midi": utils.collate_1d([torch.Tensor(s["note_midi"]) for s in samples], -1),
            "note_rest": utils.collate_1d([torch.BoolTensor(s["note_rest"]) for s in samples], True),
            "mel2note" : utils.collate_1d([torch.LongTensor(s["mel2note"]) for s in samples], 0),
            "pitch": utils.collate_1d([torch.FloatTensor(s["pitch"]) for s in samples], 0.0),
            "base_pitch" : utils.collate_1d([torch.FloatTensor(s["base_pitch"]) for s in samples], 0.0),
        }
        if self.hparams['use_spk_id']:
            batch_item["spk_id"] = torch.LongTensor([s["spk_id"] for s in samples])
        b, t, device = len(samples), batch_item["mel2note"].shape[1], batch_item["note_midi"].device
        batch_item["pitch_retake"] = random_retake_masks(b, t, device)
        return batch_item