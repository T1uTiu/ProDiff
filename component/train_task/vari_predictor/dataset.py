from typing import List
import torch
from component.train_task.base_dataset import BaseDataset
import utils

class VariPredictorDataset(BaseDataset):
    def __init__(self, prefix, shuffle, hparams, vari_type):
        super().__init__(prefix, shuffle, hparams)
        self.vari_type = vari_type

    def collater(self, samples: List[dict]):
        if len(samples) == 0:
            return {}
        batch_item = {
            "nsamples" : len(samples),
            "note_midi": utils.collate_1d([torch.Tensor(s["note_midi"]) for s in samples], -1),
            "note_rest": utils.collate_1d([torch.BoolTensor(s["note_rest"]) for s in samples], True),
            "mel2note" : utils.collate_1d([torch.LongTensor(s["mel2note"]) for s in samples], 0),
            "f0" : utils.collate_1d([torch.FloatTensor(s["f0"]) for s in samples], 0.0),
            self.vari_type: utils.collate_1d([torch.FloatTensor(s[self.vari_type]) for s in samples], 0.0),
        }
        return batch_item
    
class VoicingPredictorDataset(VariPredictorDataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__(prefix, shuffle, hparams, "voicing")
    
class BreathPredictorDataset(VariPredictorDataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__(prefix, shuffle, hparams, "breath")