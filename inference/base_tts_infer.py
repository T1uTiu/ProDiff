import json
import os
import time

import numpy as np
import torch

from modules.FastDiff.module.FastDiff_model import FastDiff
from modules.FastDiff.module.util import (compute_hyperparams_given_schedule,
                                          sampling_given_noise_schedule)
from modules.fastspeech.tts_modules import LengthRegulator
from preprocess.data_gen_utils import build_phone_encoder
from tasks.tts.dataset_utils import FastSpeechWordDataset
from utils.ckpt_utils import load_ckpt
from utils.hparams import set_hparams
from utils.pitch_utils import resample_align_curve, setuv_f0


class BaseTTSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.lr = LengthRegulator()
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_phone_encoder(self.data_dir)
        self.spk_map = self.build_spk_map()
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder, self.diffusion_hyperparams, self.noise_schedule = self.build_vocoder()

    def build_spk_map(self):
        spk_map_fn = os.path.join(self.data_dir, 'spk_map.json')
        spk_map = json.load(open(spk_map_fn, 'r'))
        return spk_map

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        raise NotImplementedError

    def run_vocoder(self, c, **kwargs):
        raise NotImplementedError

    def load_speaker_mix(self):
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
    
    def preprocess_input(self, inp):
        hparams = self.hparams

        ph = inp.get("ph_seq") # 音素
        ph_token = torch.LongTensor(self.ph_encoder.encode(ph)).to(self.device)[None, :] # [B=1, T_txt]
        
        ph_dur = torch.from_numpy(np.array(inp.get("ph_dur").split(), np.float32)).to(self.device) # 音素时长
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, ph_token == 0)  # => [B=1, T]
        
        f0_seq = resample_align_curve(
            np.array(inp.get('f0_seq').split(), np.float32),
            original_timestep=float(inp.get('f0_timestep')),
            target_timestep=self.timestep,
            align_length=mel2ph.shape[1]
        )
        f0_seq = setuv_f0(f0_seq, ph.split(), durations.cpu().numpy().squeeze(), hparams['phone_uv_set'])
        f0_seq = torch.from_numpy(f0_seq)[None, :].to(self.device) # [B=1, T_mel]

        item = {
            'ph_tokens': ph_token,
            'mel2phs': mel2ph,
            'f0_seqs': f0_seq
        }

        if hparams["use_spk_id"]:
            spk_mix_id, spk_mix_value = self.load_speaker_mix()
            item["spk_mix_id"] = spk_mix_id
            item["spk_mix_value"] = spk_mix_value
        return item


    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output