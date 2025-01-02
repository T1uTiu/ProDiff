import os

import numpy as np
from component.inferer.base import Inferer, register_inferer
from modules.ProDiff.prodiff_teacher import ProDiffTeacher
from utils.ckpt_utils import load_ckpt

@register_inferer
class ProDiffTeacherInferrer(Inferer):
    def build_model(self, ph_encoder):
        f0_stats_fn = f'{self.hparams["work_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            self.hparams['f0_mean'], self.hparams['f0_std'] = np.load(f0_stats_fn)
            self.hparams['f0_mean'] = float(self.hparams['f0_mean'])
            self.hparams['f0_std'] = float(self.hparams['f0_std'])
        model = ProDiffTeacher(ph_encoder, self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model')
        model.to(self.device)
        self.model = model

    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        f0_seq = inp['f0_seq']
        mel2ph = inp['mel2ph']
        spk_mix_embed = inp.get('spk_mix_embed', None)
        gender_mix_embed = inp.get("gender_mix_embed", None)
        lang_seq = inp.get('lang_seq', None)
        vociing = inp.get('voicing', None)
        breath = inp.get('breath', None)
        mel_out = self.model(
            ph_seq, f0=f0_seq, mel2ph=mel2ph, 
            spk_mix_embed=spk_mix_embed, gender_mix_embed=gender_mix_embed,
            lang_seq=lang_seq, 
            voicing=vociing, breath=breath,
            infer=True
        )
        if self.hparams.get('harmonic_aperiodic_seperate', False):
            return mel_out[0] + mel_out[1]
        else:
            return mel_out
    
    @staticmethod
    def category():
        return "svs"