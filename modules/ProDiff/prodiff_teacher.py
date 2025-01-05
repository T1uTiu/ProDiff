import torch
import torch.nn as nn

from modules.commons.common_layers import *
from modules.diffusion.denoise import DiffNet
from modules.diffusion.prodiff import GaussianDiffusion
from modules.fastspeech.tts_modules import FastspeechEncoder, mel2ph_to_dur
from utils.pitch_utils import f0_to_coarse

class ProDiffTeacher(nn.Module):
    def __init__(self, ph_encoder, hparams):
        super(ProDiffTeacher, self).__init__()
        self.encoder = FastspeechEncoder(
            ph_encoder=ph_encoder,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["enc_layers"],
            kernel_size=hparams["enc_ffn_kernel_size"],
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"]
        )
        self.with_dur_embed = hparams.get('use_dur_embed', True)
        if self.with_dur_embed:
            self.dur_embed = Linear(1, hparams["hidden_size"])

        self.with_spk_embed = hparams.get('use_spk_id', True)
        if self.with_spk_embed:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.with_gender_embed = hparams.get("use_gender_id", False)
        if self.with_gender_embed:
            self.gender_embed = Embedding(2, hparams['hidden_size'])

        self.with_lang_embed = hparams.get('use_lang_id', True)
        if self.with_lang_embed:
            self.lang_embed = Embedding(len(hparams["dictionary"]), hparams['hidden_size'], ph_encoder.pad())

        self.pitch_embed = Linear(1, hparams['hidden_size'])

        self.with_voicing_embed = hparams.get("use_voicing_embed", False)
        if self.with_voicing_embed:
            self.voicing_embed = Linear(1, hparams['hidden_size'])
        
        self.with_breath_embed = hparams.get("use_breath_embed", False)
        if self.with_breath_embed:
            self.breath_embed = Linear(1, hparams['hidden_size'])

        self.diffusion = GaussianDiffusion(
            out_dims=hparams["audio_num_mel_bins"],
            denoise_fn=DiffNet(
                in_dims=hparams['audio_num_mel_bins'],
                hidden_size=hparams["hidden_size"],
                residual_layers=hparams["residual_layers"],
                residual_channels=hparams["residual_channels"],
                dilation_cycle_length=hparams["dilation_cycle_length"],
            ),
            timesteps=hparams["timesteps"],
            time_scale=hparams["timescale"],
            schedule_type=hparams['schedule_type'],
            max_beta=hparams.get("max_beta", 0.02),
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )


    def add_spk_embed(self, spk_embed_id, spk_mix_embed):
        assert not (spk_embed_id is None and spk_mix_embed is None)
        if spk_mix_embed is not None:
            spk_embed = spk_mix_embed
        else:
            spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
        return spk_embed
    
    def add_gender_embed(self, gender_embed_id, gender_mix_embed):
        assert not (gender_embed_id is None and gender_mix_embed is None)
        if gender_mix_embed is not None:
            return gender_mix_embed
        return self.lang_embed(gender_embed_id)[:, None, :]

    def add_pitch(self, f0:torch.Tensor):
        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, : , None])
        return pitch_embed

    def forward(self, txt_tokens, mel2ph, f0, 
                lang_seq=None, 
                spk_embed_id=None, spk_mix_embed=None, 
                gender_embed_id=None, gender_mix_embed=None,
                voicing=None, breath=None,
                ref_mels=None,  
                infer=False, **kwargs):
        # dur embed
        if self.with_dur_embed:
            dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
            extra_embed = self.dur_embed(dur[:, :, None])
        # lang embed
        if self.with_lang_embed:
            assert lang_seq is not None, "use_lang_embed is True, lang_seq is required"
            lang_embed = self.lang_embed(lang_seq)
            extra_embed += lang_embed
        # encode
        encoder_out = self.encoder(txt_tokens, extra_embed)
        # length regulate
        condition = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(condition, 1, mel2ph_)
        # pitch
        condition += self.add_pitch(f0)
        # spk embed
        if self.with_spk_embed:
            condition += self.add_spk_embed(spk_embed_id, spk_mix_embed)
        if self.with_gender_embed:
            condition += self.add_gender_embed(gender_embed_id, gender_mix_embed)
        # variance
        variance_embeds = []
        # voicing
        if self.with_voicing_embed:
            voicing_embed = self.voicing_embed(voicing[:, :, None])
            variance_embeds.append(voicing_embed)
        # breath
        if self.with_breath_embed:
            breath_embed = self.breath_embed(breath[:, :, None])
            variance_embeds.append(breath_embed)
        if len(variance_embeds) > 0:
            condition += torch.stack(variance_embeds, dim=-1).sum(-1)
        nonpadding = (mel2ph > 0).float()[:, :, None]
        condition = condition * nonpadding
        # diffusion
        nonpadding = (mel2ph > 0).float().unsqueeze(1).unsqueeze(1)
        mel_out = self.diffusion(condition, nonpadding=nonpadding, ref_mels=ref_mels, infer=infer)
        return mel_out