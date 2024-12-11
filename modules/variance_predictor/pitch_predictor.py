import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import Linear, Embedding
from modules.diffusion.denoise import DiffNet
from modules.diffusion.prodiff import PitchDiffusion
from modules.fastspeech.tts_modules import FastspeechEncoder

class PitchPredictor(nn.Module):
    def __init__(self, ph_encoder, hparams):
        super().__init__()
        self.encoder = FastspeechEncoder(
            ph_encoder=ph_encoder, 
            hidden_size=hparams['hidden_size'], 
            num_layers=hparams['enc_layers'], 
            kernel_size=hparams['enc_ffn_kernel_size'], 
            num_heads=hparams['num_heads']
        )
        self.with_spk_embed = hparams.get('use_spk_id', True)
        if self.with_spk_embed:
            self.spk_embed = Embedding(len(hparams['datasets']), hparams['hidden_size'])
        # pitch
        self.base_f0_embed = Linear(1, hparams["hidden_size"])
        f0_hparams = hparams['f0_prediction_args']
        self.diffusion = PitchDiffusion(
            repeat_bins=f0_hparams["repeat_bins"],\
            denoise_fn=DiffNet(
                in_dims=f0_hparams["repeat_bins"],
                hidden_size=hparams["hidden_size"],
                residual_layers=f0_hparams["denoise_args"]["residual_layers"],
                residual_channels=f0_hparams["denoise_args"]["residual_channels"],
                dilation_cycle_length=f0_hparams["denoise_args"]["dilation_cycle_length"],
            ),
            timesteps=hparams["timesteps"],
            time_scale=hparams["timescale"],
            spec_min=f0_hparams["spec_min"],
            spec_max=f0_hparams["spec_max"],
            clamp_min=f0_hparams["clamp_min"],
            clamp_max=f0_hparams["clamp_max"],
        )

    def forward(self, txt_tokens, mel2ph, base_f0, f0, spk_id=None):
        encoder_out = self.encoder(txt_tokens, extra_embed=None)

        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2ph_)

        # spk
        if self.with_spk_embed:
            spk_embed = self.spk_embed(spk_id)[:, None, :]
            condition += spk_embed

        # f0
        base_f0_embed = self.base_f0_embed(base_f0[:, :, None])
        condition += base_f0_embed

        # diffusion
        nonpadding = (mel2ph > 0).float().unsqueeze(1).unsqueeze(1)
        delta_f0 = f0 - base_f0
        pitch_pred = self.diffusion(condition, nonpadding=nonpadding, ref_mels=delta_f0, infer=False)
        return pitch_pred