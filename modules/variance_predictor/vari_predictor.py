import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import Linear, Embedding
from modules.diffusion.denoise import DiffNet
from modules.diffusion.prodiff import RepeatitiveDiffusion
from modules.fastspeech.tts_modules import NoteEncoder, mel2ph_to_dur

class VariPredictor(nn.Module):
    def __init__(self, hparams, vari_type):
        super().__init__()
        vari_prediction_args = hparams[f'{vari_type}_prediction_args']
        self.encoder = NoteEncoder(
            hidden_size=vari_prediction_args["encoder_args"]['hidden_size'], 
            num_layers=vari_prediction_args["encoder_args"]['num_layers'], 
            kernel_size=vari_prediction_args["encoder_args"]['ffn_kernel_size'], 
            num_heads=vari_prediction_args["encoder_args"]['num_heads']
        )
        self.encode_out_linear = Linear(vari_prediction_args["encoder_args"]['hidden_size'], hparams['hidden_size'])

        # pitch
        self.pitch_embed = Linear(1, hparams["hidden_size"])
        # diffusion
        self.diffusion = RepeatitiveDiffusion(
            repeat_bins=vari_prediction_args["repeat_bins"],\
            denoise_fn=DiffNet(
                in_dims=vari_prediction_args["repeat_bins"],
                hidden_size=hparams["hidden_size"],
                residual_layers=vari_prediction_args["denoise_args"]["residual_layers"],
                residual_channels=vari_prediction_args["denoise_args"]["residual_channels"],
                dilation_cycle_length=vari_prediction_args["denoise_args"]["dilation_cycle_length"],
            ),
            timesteps=hparams["timesteps"],
            time_scale=hparams["timescale"],
            spec_min=vari_prediction_args["spec_min"],
            spec_max=vari_prediction_args["spec_max"],
        )

    def add_f0(self, f0: torch.Tensor):
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, : , None])
        return pitch_embed
    
    def forward(self, note_midi, note_rest, mel2note, 
                f0=None, ref_vari=None,
                infer=False):
        # encode
        note_dur = mel2ph_to_dur(mel2note, note_midi.shape[1]).float()
        encoder_out = self.encoder(note_midi, note_rest, note_dur)
        encoder_out = self.encode_out_linear(encoder_out)

        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2note_ = mel2note[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2note_)

        # f0
        condition += self.add_f0(f0)

        # diffusion
        nonpadding = (mel2note > 0).float().unsqueeze(1).unsqueeze(1)
        vari_pred = self.diffusion(condition, nonpadding=nonpadding, ref_mels=ref_vari, infer=infer)
        return vari_pred

class VoicingPredictor(VariPredictor):
    def __init__(self, hparams):
        super().__init__(hparams, "voicing")

class BreathPredictor(VariPredictor):
    def __init__(self, hparams):
        super().__init__(hparams, "breath")