import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import Embedding, Linear
from modules.diffusion.denoise import DiffNet
from modules.diffusion.prodiff import MultiVariDiffusion
from modules.fastspeech.tts_modules import NoteEncoder, FastspeechEncoder, mel2ph_to_dur

class VariPredictor(nn.Module):
    def __init__(self, vocab_size, hparams):
        super().__init__()
        self.with_dur_embed = hparams.get('use_dur_embed', True)
        if self.with_dur_embed:
            self.dur_embed = Linear(1, hparams["hidden_size"])
        self.encoder = FastspeechEncoder(
            vocab_size=vocab_size,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["enc_layers"],
            kernel_size=hparams["enc_ffn_kernel_size"],
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"]
        )
        vari_prediction_args = hparams['vari_prediction_args']
        self.note_encoder = NoteEncoder(
            hidden_size=vari_prediction_args["encoder_args"]['hidden_size'], 
            num_layers=vari_prediction_args["encoder_args"]['num_layers'], 
            kernel_size=vari_prediction_args["encoder_args"]['ffn_kernel_size'], 
            num_heads=vari_prediction_args["encoder_args"]['num_heads']
        )
        self.note_encode_out_linear = Linear(vari_prediction_args["encoder_args"]['hidden_size'], hparams['hidden_size'])
        # spk
        self.with_spk_embed = hparams.get('use_spk_id', True)
        if self.with_spk_embed:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])
        # pitch
        self.pitch_embed = Linear(1, hparams["hidden_size"])
        # vari
        self.pred_voicing = hparams.get("use_voicing_embed", False)
        self.pred_breath = hparams.get("use_breath_embed", False)
        self.pred_tension = hparams.get("use_tension_embed", False)
        self.variance_list = []
        vari_spec_range = [[], []] #[[min], [max]]
        vari_clamp_range = [[], []] #[[min], [max]]
        if self.pred_voicing:
            self.variance_list.append("voicing")
            vari_spec_range[0].append(hparams["voicing_db_min"])
            vari_spec_range[1].append(hparams["voicing_db_max"])
            vari_clamp_range[0].append(hparams["voicing_db_min"])
            vari_clamp_range[1].append(hparams["voicing_db_max"])
        if self.pred_breath:
            self.variance_list.append("breath")
            vari_spec_range[0].append(hparams["breath_db_min"])
            vari_spec_range[1].append(hparams["breath_db_max"])
            vari_clamp_range[0].append(hparams["breath_db_min"])
            vari_clamp_range[1].append(hparams["breath_db_max"])
        if self.pred_tension:
            self.variance_list.append("tension")
            vari_spec_range[0].append(hparams["tension_logit_min"])
            vari_spec_range[1].append(hparams["tension_logit_max"])
            vari_clamp_range[0].append(hparams["tension_logit_min"])
            vari_clamp_range[1].append(hparams["tension_logit_max"])
        repeat_bins = vari_prediction_args["repeat_bins"] // len(self.variance_list)
        self.diffusion = MultiVariDiffusion(
            repeat_bins=repeat_bins,
            denoise_fn=DiffNet(
                in_dims=repeat_bins,
                hidden_size=hparams["hidden_size"],
                residual_layers=vari_prediction_args["denoise_args"]["residual_layers"],
                residual_channels=vari_prediction_args["denoise_args"]["residual_channels"],
                dilation_cycle_length=vari_prediction_args["denoise_args"]["dilation_cycle_length"],
            ),
            timesteps=vari_prediction_args["timesteps"],
            time_scale=vari_prediction_args["timescale"],
            spec_min=vari_spec_range[0],
            spec_max=vari_spec_range[1],
            clamp_min=vari_clamp_range[0],
            clamp_max=vari_clamp_range[1],
        )

    def add_f0(self, f0: torch.Tensor):
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, : , None])
        return pitch_embed
    
    def add_spk_embed(self, spk_embed_id):
        spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
        return spk_embed

    def collect_vari_input(self, **kwargs):
        return [kwargs.get(name) for name in self.variance_list]

    def collect_vari_output(self, variacne_pred: list):
        return {name: pred for name, pred in zip(self.variance_list, variacne_pred)}

    def forward(self, 
                txt_tokens, mel2ph,
                note_midi, note_rest, mel2note, 
                f0, 
                spk_embed_id=None,
                infer=False, **kwargs):
        # dur embed
        if self.with_dur_embed:
            dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
            extra_embed = self.dur_embed(dur[:, :, None])
        # encode
        encoder_out = self.encoder(txt_tokens, extra_embed)
        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2ph_)

        # note_encode
        note_dur = mel2ph_to_dur(mel2note, note_midi.shape[1]).float()
        note_encoder_out = self.note_encoder(note_midi, note_rest, note_dur)
        note_encoder_out = self.note_encode_out_linear(note_encoder_out)
        # length regulate
        note_encoder_out = F.pad(note_encoder_out, [0, 0, 1, 0])
        mel2note_ = mel2note[..., None].repeat([1, 1, note_encoder_out.shape[-1]])
        note_condition = torch.gather(note_encoder_out, 1, mel2note_)
        condition += note_condition

        # f0
        condition += self.add_f0(f0)
        # spk embed
        if self.with_spk_embed:
            condition += self.add_spk_embed(spk_embed_id)
        # diffusion
        nonpadding = (mel2note > 0).float().unsqueeze(1).unsqueeze(1)
        vari_input = self.collect_vari_input(**kwargs)
        vari_pred = self.diffusion(condition, nonpadding=nonpadding, ref_mels=vari_input, infer=infer)
        if infer:
            return self.collect_vari_output(vari_pred)
        return vari_pred

class VoicingPredictor(VariPredictor):
    def __init__(self, hparams):
        super().__init__(hparams, "voicing")

class BreathPredictor(VariPredictor):
    def __init__(self, hparams):
        super().__init__(hparams, "breath")