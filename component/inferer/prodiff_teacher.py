import os
from typing import Dict

import numpy as np
from component.inferer.base import Inferer, register_inferer
from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from usr.diff.net import DiffNet
from utils.ckpt_utils import load_ckpt

@register_inferer
class ProDiffTeacherInferrer(Inferer):
    def build_model(self, ph_encoder):
        f0_stats_fn = f'{self.hparams["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            self.hparams['f0_mean'], self.hparams['f0_std'] = np.load(f0_stats_fn)
            self.hparams['f0_mean'] = float(self.hparams['f0_mean'])
            self.hparams['f0_std'] = float(self.hparams['f0_std'])
        model = GaussianDiffusion(
            phone_encoder=ph_encoder,
            out_dims=self.hparams['audio_num_mel_bins'], denoise_fn=DiffNet(self.hparams['audio_num_mel_bins']),
            timesteps=self.hparams['timesteps'],
            loss_type=self.hparams['diff_loss_type'],
            spec_min=self.hparams['spec_min'], spec_max=self.hparams['spec_max'],
        )
        model.eval()
        load_ckpt(model, self.hparams['work_dir'], 'model')
        model.to(self.device)
        self.model = model

    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        f0_seq = inp['f0_seq']
        mel2ph = inp['mel2ph']
        spk_mix_embed = inp.get('spk_mix_embed', None)
        lang_seq = inp.get('lang_seq', None)
        return self.model(ph_seq, f0=f0_seq, mel2ph=mel2ph, spk_mix_embed=spk_mix_embed, lang_seq=lang_seq, infer=True)