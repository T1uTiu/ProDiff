from typing import List
import torch
from component.train_task.base_dataset import BaseDataset
import utils

class VariPredictorDataset(BaseDataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__(prefix, shuffle, hparams)

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
            "f0" : utils.collate_1d([torch.FloatTensor(s["f0"]) for s in samples], 0.0),
        }
        if self.hparams['use_spk_id']:
            batch_item["spk_id"] = torch.LongTensor([s["spk_id"] for s in samples])
        if self.hparams["use_voicing_embed"]:
            batch_item["voicing"] = utils.collate_1d([torch.FloatTensor(s["voicing"]) for s in samples], 0.0)
        if self.hparams["use_breath_embed"]:
            batch_item["breath"] = utils.collate_1d([torch.FloatTensor(s["breath"]) for s in samples], 0.0)
        if self.hparams["use_tension_embed"]:
            batch_item["tension"] = utils.collate_1d([torch.FloatTensor(s["tension"]) for s in samples], 0.0)
        return batch_item
    