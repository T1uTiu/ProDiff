import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import Linear, Embedding
from modules.decoder.wavenet import WaveNet
from modules.diffusion.reflow import PitchRectifiedFlow
from modules.fastspeech.tts_modules import NoteEncoder, FastspeechEncoder, mel2ph_to_dur

class PitchPredictor(nn.Module):
    def __init__(self, vocab_size, hparams):
        super().__init__()
        self.encoder = FastspeechEncoder(
            vocab_size=vocab_size+1,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["enc_layers"],
            kernel_size=hparams["enc_ffn_kernel_size"],
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"]
        )
        self.with_dur_embed = hparams.get('use_dur_embed', True)
        if self.with_dur_embed:
            self.dur_embed = Linear(1, hparams["hidden_size"])
        f0_prediction_args = hparams['f0_prediction_args']
        self.note_encoder = NoteEncoder(
            hidden_size=f0_prediction_args["encoder_args"]['hidden_size'], 
            num_layers=f0_prediction_args["encoder_args"]['num_layers'], 
            kernel_size=f0_prediction_args["encoder_args"]['ffn_kernel_size'], 
            num_heads=f0_prediction_args["encoder_args"]['num_heads']
        )
        self.note_encode_out_linear = Linear(f0_prediction_args["encoder_args"]['hidden_size'], hparams['hidden_size'])

        self.with_spk_embed = hparams.get('use_spk_id', True)
        if self.with_spk_embed:
            self.spk_embed = Embedding(len(hparams['datasets']), hparams['hidden_size'])

        # pitch
        self.delta_pitch_embed = Linear(1, hparams["hidden_size"])
        self.pitch_retake_embed = Embedding(2, hparams['hidden_size'])
        self.sample_steps = hparams["sampling_steps"]
        self.diffusion = PitchRectifiedFlow(
            repeat_bins=f0_prediction_args["repeat_bins"],\
            denoise_fn=WaveNet(
                in_dims=f0_prediction_args["repeat_bins"],
                hidden_size=hparams["hidden_size"],
                residual_layers=f0_prediction_args["denoise_args"]["residual_layers"],
                residual_channels=f0_prediction_args["denoise_args"]["residual_channels"],
                dilation_cycle_length=f0_prediction_args["denoise_args"]["dilation_cycle_length"],
            ),
            time_scale=f0_prediction_args["timescale"],
            sampling_algorithm=hparams["sampling_algorithm"],
            spec_min=f0_prediction_args["spec_min"],
            spec_max=f0_prediction_args["spec_max"],
            clamp_min=f0_prediction_args["clamp_min"],
            clamp_max=f0_prediction_args["clamp_max"],
        )

    def forward(
            self, txt_tokens, mel2ph,
            note_midi, note_rest, mel2note, 
            base_pitch, pitch=None, 
            pitch_retake=None, pitch_expr=None,
            spk_id=None, infer=False
        ):
        # dur embed
        if self.with_dur_embed:
            dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
            extra_embed = self.dur_embed(dur[:, :, None])
        # ph encode
        encoder_out = self.encoder(txt_tokens, extra_embed)
        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2ph_)
        # encode
        note_dur = mel2ph_to_dur(mel2note, note_midi.shape[1]).float()
        note_encoder_out = self.note_encoder(note_midi, note_rest, note_dur)
        note_encoder_out = self.note_encode_out_linear(note_encoder_out)
        # length regulate
        note_encoder_out = F.pad(note_encoder_out, [0, 0, 1, 0])
        mel2note_ = mel2note[..., None].repeat([1, 1, note_encoder_out.shape[-1]])
        note_condition = torch.gather(note_encoder_out, 1, mel2note_)
        condition += note_condition

        # spk
        if self.with_spk_embed:
            spk_embed = self.spk_embed(spk_id)[:, None, :]
            condition += spk_embed

        # pitch retake
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

        # delta_pitch
        if is_pitch_retake:
            delta_pitch = (pitch - base_pitch) * ~pitch_retake
        else:
            delta_pitch = torch.zeros_like(base_pitch)
        delta_pitch_embed = self.delta_pitch_embed(delta_pitch[:, :, None])
        condition += delta_pitch_embed

        # diffusion
        if not infer:
            pitch_pred = self.diffusion(condition, pitch-base_pitch, infer=False)
        else:
            pitch_pred = self.diffusion(condition, infer_step=self.sample_steps, infer=True)
        return pitch_pred
