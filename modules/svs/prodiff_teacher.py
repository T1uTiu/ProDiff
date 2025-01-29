import torch
import torch.nn as nn

from modules.commons.common_layers import *
from modules.decoder.wavenet import WaveNet
from modules.diffusion.prodiff import GaussianDiffusion
from modules.diffusion.reflow import RectifiedFlow
from modules.fastspeech.tts_modules import FastspeechEncoder, mel2ph_to_dur

class ProDiffTeacher(nn.Module):
    def __init__(self, vocab_size, hparams):
        super(ProDiffTeacher, self).__init__()
        self.encoder = FastspeechEncoder(
            vocab_size=vocab_size,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["enc_layers"],
            kernel_size=hparams["enc_ffn_kernel_size"],
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"],
            rel_pos=hparams.get("rel_pos", False),
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
            self.lang_embed = Embedding(len(hparams["dictionary"]), hparams['hidden_size'], 0)

        self.pitch_embed = Linear(1, hparams['hidden_size'])

        self.with_voicing_embed = hparams.get("use_voicing_embed", False)
        if self.with_voicing_embed:
            self.voicing_embed = Linear(1, hparams['hidden_size'])
        
        self.with_breath_embed = hparams.get("use_breath_embed", False)
        if self.with_breath_embed:
            self.breath_embed = Linear(1, hparams['hidden_size'])

        self.diffusion_type = hparams.get("diff_type", "prodiff")
        if self.diffusion_type == "prodiff":
            self.diffusion = GaussianDiffusion(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
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
        elif self.diffusion_type == "reflow":
            self.diffusion = RectifiedFlow(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
                    in_dims=hparams['audio_num_mel_bins'],
                    hidden_size=hparams["hidden_size"],
                    residual_layers=hparams["residual_layers"],
                    residual_channels=hparams["residual_channels"],
                    dilation_cycle_length=hparams["dilation_cycle_length"],
                ),
                time_scale=hparams["timescale"],
                num_features=1,
                sampling_algorithm=hparams.get("sampling_algorithm", "euler"),
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

    def forward_condition(
            self, txt_tokens, mel2ph, f0, 
            lang_seq=None, 
            spk_embed_id=None, spk_mix_embed=None, 
            gender_embed_id=None, gender_mix_embed=None,
            voicing=None, breath=None
        ):
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
        return condition
        
    def forward(
            self, txt_tokens, mel2ph, f0, 
            lang_seq=None, 
            spk_embed_id=None, spk_mix_embed=None, 
            gender_embed_id=None, gender_mix_embed=None,
            voicing=None, breath=None,
            ref_mels=None, infer=False
        ):
        condition = self.forward_condition(
            txt_tokens, mel2ph, f0, 
            lang_seq=lang_seq, 
            spk_embed_id=spk_embed_id, spk_mix_embed=spk_mix_embed,
            gender_embed_id=gender_embed_id, gender_mix_embed=gender_mix_embed,
            voicing=voicing, breath=breath
        )
        # diffusion
        output = self.diffusion(condition, gt_spec=ref_mels, infer=infer)
        return output


class ProDiff(nn.Module):
    def __init__(self, vocab_size, hparams):
        super().__init__(vocab_size, hparams)
        self.mel_bins = hparams["audio_num_mel_bins"]
        self.teacher = ProDiffTeacher(vocab_size, hparams)
        self.diffusion_type = hparams.get("diff_type", "prodiff")
        assert self.diffusion_type in ["prodiff"]
        if self.diffusion_type == "prodiff":
            self.diffusion = GaussianDiffusion(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
                    in_dims=hparams['audio_num_mel_bins'],
                    hidden_size=hparams["hidden_size"],
                    residual_layers=hparams["residual_layers"],
                    residual_channels=hparams["residual_channels"],
                    dilation_cycle_length=hparams["dilation_cycle_length"],
                ),
                timesteps=1,
                time_scale=hparams["timescale"],
                schedule_type=hparams['schedule_type'],
                max_beta=hparams.get("max_beta", 0.02),
                spec_min=hparams["spec_min"],
                spec_max=hparams["spec_max"],
            )
        elif self.diffusion_type == "reflow":
            self.diffusion = RectifiedFlow(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
                    in_dims=hparams['audio_num_mel_bins'],
                    hidden_size=hparams["hidden_size"],
                    residual_layers=hparams["residual_layers"],
                    residual_channels=hparams["residual_channels"],
                    dilation_cycle_length=hparams["dilation_cycle_length"],
                ),
                time_scale=hparams["timescale"],
                num_features=1,
                sampling_algorithm=hparams.get("sampling_algorithm", "euler"),
                spec_min=hparams["spec_min"],
                spec_max=hparams["spec_max"],
            )
            
    def forward(
            self, txt_tokens, mel2ph, f0, 
            lang_seq=None, 
            spk_embed_id=None, spk_mix_embed=None, 
            gender_embed_id=None, gender_mix_embed=None,
            voicing=None, breath=None,
            ref_mels=None, infer=False
        ):
        with torch.no_grad():
            condition = self.teacher.forward_condition(
                txt_tokens, mel2ph, f0, 
                lang_seq=lang_seq, 
                spk_embed_id=spk_embed_id, spk_mix_embed=spk_mix_embed,
                gender_embed_id=gender_embed_id, gender_mix_embed=gender_mix_embed,
                voicing=voicing, breath=breath
            )
            b, device = condition.shape[0], condition.device
            x_T = torch.randn(b, 1, self.mel_bins, condition.shape[2], device=device)
            if not infer:
                x_0 = self.teacher.diffusion(condition, x_T, infer=True)
                x_0 = x_0.transpose(-2, -1)[:, None, :, :]
        if not infer:
            output = self.diffusion(condition, x_T, gt_spec=x_0, infer=False)
        else:
            output = self.diffusion(condition, x_T, infer=True)
        return output