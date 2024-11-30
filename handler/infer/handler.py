import json
import os
import time
from typing import List

import numpy as np
import torch

from component.inferer.base import get_inferer_cls
from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from modules.fastspeech.tts_modules import LengthRegulator
from usr.diff.net import DiffNet
from utils.audio import cross_fade, save_wav
from utils.ckpt_utils import load_ckpt
from utils.data_gen_utils import build_phone_encoder
from utils.pitch_utils import resample_align_curve, shift_pitch
from utils.text_encoder import TokenTextEncoder
from vocoders.base_vocoder import get_vocoder_cls


class InferHandler:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_data_dir = hparams['binary_data_dir']

        hop_size = hparams['hop_size']
        self.audio_sample_rate = hparams['audio_sample_rate']
        self.timestep = hop_size / self.audio_sample_rate

        self.ph2merged, self.ph_encoder = self.build_phone_encoder()
        self.spk_map = self.build_spk_map()
        self.lang_map = self.build_lang_map()

        self.lr = LengthRegulator()
        self.inferer = get_inferer_cls(hparams)(hparams)
        self.inferer.build_model(self.ph_encoder)
        # self.model = self.build_model()
        self.vocoder = self.build_vocoder()

    def build_phone_encoder(self):
        ph2merged = {}
        if self.hparams["merged_phoneme_dict"] is not None and self.hparams["merged_phoneme_dict"] != "":
            fn = f"{self.hparams['binary_data_dir']}/{self.hparams['merged_phoneme_dict']}"
            f = open(fn, 'r')
            merge_dict = json.load(f)
            for merged, phs in merge_dict.items():
                for ph in phs:
                    ph2merged[ph] = merged
            f.close()
        ph_list_fn = os.path.join(self.binary_data_dir, 'phone_set.json')
        with open(ph_list_fn, 'r') as f:
            ph_list = json.load(f)
        return ph2merged, TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")

    def build_spk_map(self):
        spk_map_fn = os.path.join(self.binary_data_dir, 'spk_map.json')
        assert os.path.exists(spk_map_fn), f"Speaker map file {spk_map_fn} not found"
        with open(spk_map_fn, 'r') as f:
            spk_map = json.load(f)
        return spk_map
    
    def build_lang_map(self):
        lang_map_fn = os.path.join(self.binary_data_dir, 'lang_map.json')
        assert os.path.exists(lang_map_fn), f"Language map file {lang_map_fn} not found"
        with open(lang_map_fn, 'r') as f:
            lang_map = json.load(f)
        return lang_map

    def build_vocoder(self):
        vocoder = get_vocoder_cls(self.hparams)()
        vocoder.to_device(self.device)
        return vocoder

    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def build_model(self):
        f0_stats_fn = f'{self.binary_data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            self.hparams['f0_mean'], self.hparams['f0_std'] = np.load(f0_stats_fn)
            self.hparams['f0_mean'] = float(self.hparams['f0_mean'])
            self.hparams['f0_std'] = float(self.hparams['f0_std'])
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=self.hparams['audio_num_mel_bins'], denoise_fn=DiffNet(self.hparams['audio_num_mel_bins']),
            timesteps=self.hparams['timesteps'],
            loss_type=self.hparams['diff_loss_type'],
            spec_min=self.hparams['spec_min'], spec_max=self.hparams['spec_max'],
        )
        model.eval()
        load_ckpt(model, self.hparams['work_dir'], 'model')
        model.to(self.device)
        return model

    def get_speaker_mix(self):
        hparams = self.hparams
        spk_name = hparams['spk_name'] # "spk0:0.5|spk1:0.5 ..."
        if spk_name == '':
            # Get the first speaker
            spk_mix_map = {self.spk_map.keys()[0]: 1.0}
        else:
            spk_mix_map = dict([x.split(':') for x in spk_name.split('|')])
            for k in spk_mix_map:
                spk_mix_map[k] = float(spk_mix_map[k])
        spk_mix_id_list = []
        spk_mix_value_list = []
        for name, value in spk_mix_map.items():
            assert name in self.spk_map, f"Speaker name {name} not found in spk_map"
            spk_mix_id_list.append(self.spk_map[name])
            spk_mix_value_list.append(value)
        spk_mix_id = torch.LongTensor(spk_mix_id_list).to(self.device)[None, None]
        spk_mix_value = torch.FloatTensor(spk_mix_value_list).to(self.device)[None, None]
        spk_mix_value_sum = spk_mix_value.sum()
        spk_mix_value /= spk_mix_value_sum # Normalize
        return spk_mix_id, spk_mix_value

    def get_ph_text(self, ph, lang):
        ph = f"{ph}/{lang}"
        return self.ph2merged.get(ph, ph)

    def infer(self, segment: dict):
        lang = segment.get("lang", "zh")
        ph_text_seq = [self.get_ph_text(ph, lang) for ph in segment["ph_seq"].split()]

        if self.hparams["use_lang_id"]:
            lang_seq = torch.LongTensor(
                [self.lang_map[lang]] * len(ph_text_seq)
            ).to(self.device)[None, :] # [B=1, T_txt]

        ph_token_seq = torch.LongTensor(
            self.ph_encoder.encode(ph_text_seq)
        ).to(self.device)[None, :] # [B=1, T_txt]

        ph_dur = torch.from_numpy(np.array(segment["ph_dur"].split(), np.float32)).to(self.device) # 音素时长
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, ph_token_seq == 0)  # => [B=1, T]

        f0_seq = resample_align_curve(
            np.array(segment['f0_seq'].split(), np.float32),
            original_timestep=float(segment['f0_timestep']),
            target_timestep=self.timestep,
            align_length=mel2ph.shape[1]
        )
        keyshift = segment.get("keyshift", 0)
        if keyshift != 0:
            f0_seq = shift_pitch(f0_seq, keyshift)
        f0_seq = torch.from_numpy(f0_seq)[None, :].to(self.device) # [B=1, T_mel]

        spk_mix_embed = None
        if self.hparams["use_spk_id"]:
            spk_mix_id, spk_mix_value = self.get_speaker_mix()
            spk_mix_embed = torch.sum(
                self.inferer.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3), 
                dim=2, keepdim=False
            )
        
        with torch.no_grad():
            start_time = time.time()
            output = self.inferer.run_model(ph_seq=ph_token_seq, f0_seq=f0_seq, mel2ph=mel2ph, spk_mix_embed=spk_mix_embed, lang_seq=lang_seq, infer=True)
            print(f"Inference Time: {time.time() - start_time}")
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out, f0=f0_seq)
        wav_out = wav_out.squeeze().cpu().numpy()
        return wav_out
        

    def handle(self, proj: List[dict] = None, proj_fn=None, lang=None, keyshift=0):
        if proj is None:
            with open(proj_fn, 'r', encoding='utf-8') as f:
                proj = json.load(f)
        result = np.zeros(0)
        total_length = 0
        for segment in proj:
            segment.setdefault('lang', lang)
            segment.setdefault("keyshift", int(keyshift))
            out = self.infer(segment)
            offset = round(segment.get('offset', 0) * self.audio_sample_rate) - total_length
            if offset >= 0:
                result = np.append(result, np.zeros(offset))
                result = np.append(result, out)
            else:
                result = cross_fade(result, out, total_length + offset)
            total_length += offset + out.shape[0]
        title = proj_fn.split('/')[-1].split('.')[0]
        out_fn = f'infer_out/{title}【{self.hparams["exp_name"]}】.wav'
        save_wav(result, out_fn, self.audio_sample_rate)