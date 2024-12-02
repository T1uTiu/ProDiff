import torch.nn as nn

from modules.diffusion.teacher import GaussianDiffusion
from modules.fastspeech.fs2 import FastSpeech2
from usr.diff.net import DiffNet

class ProDiffTeacher(nn.Module):
    def __init__(self, ph_encoder, hparams):
        self.fs = FastSpeech2(
            ph_encoder=ph_encoder,
            hidden_size=hparams["hidden_size"],
            enc_layers=hparams["enc_layers"],
            enc_ffn_kernel_size=hparams["enc_ffn_kernel_size"],
            num_heads=hparams["num_heads"]
        )
        self.diffusion = GaussianDiffusion(
            out_dims=hparams["audio_num_mel_bins"],
            denoise_fn=DiffNet(hparams['audio_num_mel_bins']),
            timesteps=hparams["timestep"],
            time_scale=hparams["timescale"],
            loss_type=hparams["diff_loss_type"],
            schedule_mode=hparams['schedule_type'],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
            keep_bins=hparams["keep_bins"]
        )

    def forward(self, txt_tokens, mel2ph, f0, 
                lang_seq=None, spk_embed_id=None, spk_mix_embed=None, 
                ref_mels=None, infer=False, **kwargs):
        condition = self.fs(txt_tokens, mel2ph, f0,
                            lang_seq=lang_seq, spk_embed_id=spk_embed_id, spk_mix_embed=spk_mix_embed)
        mel_out = self.diffusion(condition, mel2ph, ref_mels=ref_mels, infer=infer)
        return mel_out