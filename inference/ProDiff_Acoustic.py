import sys, os
sys.path.append(os.getcwd())

from vocoders.base_vocoder import VOCODERS
import torch
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from inference.base_tts_infer import BaseTTSInfer
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff import GaussianDiffusion
from usr.diff.net import DiffNet
import os
import numpy as np
from functools import partial

class ProDiffInfer(BaseTTSInfer):
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
        
        checkpoint = torch.load(hparams['teacher_ckpt'], map_location='cpu')["state_dict"]['model']
        teacher_timesteps = int(checkpoint['timesteps'].item())
        teacher_timescales = int(checkpoint['timescale'].item())
        student_timesteps = teacher_timesteps // 2
        student_timescales = teacher_timescales * 2
        to_torch = partial(torch.tensor, dtype=torch.float32)
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DiffNet(hparams['audio_num_mel_bins']),
            teacher_steps=teacher_timesteps,
            timesteps=hparams['timesteps'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        model.register_buffer('timesteps', to_torch(student_timesteps))      # beta
        model.register_buffer('timescale', to_torch(student_timescales))      # beta
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        return model

    def forward_model(self, inp):
        ph_tokens = inp['ph_tokens']  # [B, T_t]
        mel2phs = inp['mel2phs']  # [B, T_t]
        f0_seqs = inp['f0_seqs']  # [B, frame]
        with torch.no_grad():
            output = self.model(ph_tokens, f0=f0_seqs, mel2ph=mel2phs, infer=True)
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out, f0=f0_seqs)
        wav_out = wav_out.squeeze().cpu().numpy()
        return wav_out


if __name__ == '__main__':
    ProDiffInfer.example_run()
