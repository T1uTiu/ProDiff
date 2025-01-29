import json
import os
from typing import List
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
from modules.svs.prodiff_teacher import ProDiffTeacher
import utils
from component.train_task.base_dataset import BaseDataset
from utils.ckpt_utils import load_ckpt
from utils.text_encoder import TokenTextEncoder


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

        if self.hparams["use_tension_embed"]:
            batch_item["tension"] = utils.collate_1d([torch.FloatTensor(s["tension"]) for s in samples], 0.0)
        
        return batch_item

class SVSRectifiedDataset(SVSDataset):
    def __init__(self, prefix, shuffle, hparams):
        super().__init__(prefix, shuffle, hparams)
        # load teacher model
        self.mel_bins = hparams["audio_num_mel_bins"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ph_map_fn = os.path.join(self.data_dir, 'phone_set.json')
        with open(ph_map_fn, 'r') as f:
            ph_map = json.load(f)
        ph_list = list(sorted(set(ph_map.values())))
        ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov='SP')
        teacher_ckpt = hparams.get["teacher_ckpt"]
        self.teacher = ProDiffTeacher(len(ph_encoder), hparams)
        load_ckpt(self.teacher, teacher_ckpt, "model")
        self.teacher.eval()
        self.teacher.to(self.device)

    def collater(self, samples: List[dict]):
        batch_item = super().collater(samples)
        ph_seq = batch_item["ph_seq"]  # [B, T_t]
        mel2ph = batch_item["mel2ph"]
        f0 = batch_item["f0"]
        spk_embed_id = batch_item.get("spk_id", None)
        gender_embed_id = batch_item.get("gender_id", None)
        lang_seq = batch_item.get("lang_seq", None)
        voicing = batch_item.get("voicing", None)
        breath = batch_item.get("breath", None)
        with torch.no_grad():
            condition = self.teacher.forward_condition(
                ph_seq, mel2ph, f0,
                lang_seq=lang_seq,
                spk_embed_id=spk_embed_id, gender_embed_id=gender_embed_id,
                voicing=voicing, breath=breath
            )
            b, device = condition.shape[0], condition.device
            x_T = torch.randn(b, 1, self.mel_bins, condition.shape[1], device=device)
            x_0 = self.teacher.diffusion(condition, x_T, infer=True)
            x_0 = x_0.transpose(-2, -1)[:, None, :, :]
            batch_item["condition"] = condition
            batch_item["x_T"] = x_T
            batch_item["x_0"] = x_0
        return batch_item
