import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import Linear, Embedding
from modules.diffusion.denoise import DiffNet
from modules.diffusion.prodiff import PitchDiffusion
from modules.fastspeech.tts_modules import NoteEncoder, mel2ph_to_dur

class PitchPredictor(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        f0_prediction_args = hparams['f0_prediction_args']
        self.encoder = NoteEncoder(
            hidden_size=f0_prediction_args["encoder_args"]['hidden_size'], 
            num_layers=f0_prediction_args["encoder_args"]['num_layers'], 
            kernel_size=f0_prediction_args["encoder_args"]['ffn_kernel_size'], 
            num_heads=f0_prediction_args["encoder_args"]['num_heads']
        )
        self.encode_out_linear = Linear(f0_prediction_args["encoder_args"]['hidden_size'], hparams['hidden_size'])

        self.with_spk_embed = hparams.get('use_spk_id', True)
        if self.with_spk_embed:
            self.spk_embed = Embedding(len(hparams['datasets']), hparams['hidden_size'])

        # pitch
        self.delta_pitch_embed = Linear(1, hparams["hidden_size"])
        self.pitch_retake_embed = Embedding(2, hparams['hidden_size'])
        self.diffusion = PitchDiffusion(
            repeat_bins=f0_prediction_args["repeat_bins"],\
            denoise_fn=DiffNet(
                in_dims=f0_prediction_args["repeat_bins"],
                hidden_size=hparams["hidden_size"],
                residual_layers=f0_prediction_args["denoise_args"]["residual_layers"],
                residual_channels=f0_prediction_args["denoise_args"]["residual_channels"],
                dilation_cycle_length=f0_prediction_args["denoise_args"]["dilation_cycle_length"],
            ),
            timesteps=hparams["timesteps"],
            time_scale=hparams["timescale"],
            spec_min=f0_prediction_args["spec_min"],
            spec_max=f0_prediction_args["spec_max"],
            clamp_min=f0_prediction_args["clamp_min"],
            clamp_max=f0_prediction_args["clamp_max"],
        )

    def forward(self, note_midi, note_rest, mel2note, 
                base_f0, f0=None, 
                pitch_retake=None, pitch_expr=None,
                spk_id=None, infer=False):
        # check params
        assert not infer and pitch_retake is not None
        assert not infer and f0 is not None
        # encode
        note_dur = mel2ph_to_dur(mel2note, note_midi.shape[1]).float()
        encoder_out = self.encoder(note_midi, note_rest, note_dur)
        encoder_out = self.encode_out_linear(encoder_out)

        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2note_ = mel2note[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2note_)

        # spk
        if self.with_spk_embed:
            spk_embed = self.spk_embed(spk_id)[:, None, :]
            condition += spk_embed

        # f0
        is_pitch_retake = pitch_retake is not None
        if not is_pitch_retake:
            pitch_retake = torch.ones_like(mel2note, dtype=torch.long)

        if pitch_expr is None:
            pitch_retake_embed = self.pitch_retake_embed(pitch_retake.long())
        else:
            retake_true_embed = self.pitch_retake_embed(
                torch.ones(1, 1, dtype=torch.long, device=note_midi.device)
            )
            retake_false_embed = self.pitch_retake_embed(
                torch.zeros(1, 1, dtype=torch.long, device=note_midi.device)
            )
            pitch_expr = (pitch_expr * pitch_retake)[:, :, None]
            pitch_retake_embed = retake_true_embed * pitch_expr + retake_false_embed * (1 - pitch_expr)
        condition += pitch_retake_embed
        if is_pitch_retake:
            delta_pitch = (f0 - base_f0) * ~pitch_retake
        else:
            delta_pitch = torch.zeros_like(base_f0)
        delta_pitch_embed = self.delta_pitch_embed(delta_pitch[:, :, None])
        condition += delta_pitch_embed

        # diffusion
        nonpadding = (mel2note > 0).float().unsqueeze(1).unsqueeze(1)
        if not infer:
            pitch_pred = self.diffusion(condition, nonpadding=nonpadding, ref_mels=f0-base_f0, infer=infer)
        else:
            pitch_pred = self.diffusion(condition, nonpadding=nonpadding, infer=infer)
        return pitch_pred
