import sys, os
sys.path.append(os.getcwd())

import torch
from inference.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from usr.diff.net import DiffNet
import os
import numpy as np
from vocoders.base_vocoder import VOCODERS

class ProDiffTeacherInfer(BaseTTSInfer):
    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        vocoder.to_device(self.device)
        return vocoder, None, None
    
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def build_model(self):
        f0_stats_fn = f'{hparams["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DiffNet(hparams['audio_num_mel_bins']),
            timesteps=hparams['timesteps'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        ph_tokens = sample['ph_tokens']  # [B, T_t]
        mel2phs = sample['mel2phs']  # [B, T_t]
        f0_seqs = sample['f0_seqs']  # [B, frame]
        with torch.no_grad():
            output = self.model(ph_tokens, f0=f0_seqs, mel2ph=mel2phs, infer=True)
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out, f0=f0_seqs)
        wav_out = wav_out.squeeze().cpu().numpy()
        return wav_out


if __name__ == '__main__':
    ProDiffTeacherInfer.example_run()
