import json
import os
import time

import torch

from data_gen.tts.data_gen_utils import build_phone_encoder
from modules.fastspeech.tts_modules import LengthRegulator
from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
import numpy as np
from modules.FastDiff.module.util import compute_hyperparams_given_schedule, sampling_given_noise_schedule

import os

import torch

from modules.FastDiff.module.FastDiff_model import FastDiff
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
        self.ph_encoder = self.build_phone_encoder()
        self.spk_map = self.build_spk_map()
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder, self.diffusion_hyperparams, self.noise_schedule = self.build_vocoder()

    def build_phone_encoder(self):
        return build_phone_encoder(self.data_dir)

    def build_spk_map(self):
        spk_map_fn = os.path.join(self.data_dir, 'spk_map.json')
        spk_map = json.load(open(spk_map_fn, 'r'))
        return spk_map

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = FastDiff(audio_channels=config['audio_channels'],
                 inner_channels=config['inner_channels'],
                 cond_channels=config['cond_channels'],
                 upsample_ratios=config['upsample_ratios'],
                 lvc_layers_each_block=config['lvc_layers_each_block'],
                 lvc_kernel_size=config['lvc_kernel_size'],
                 kpnet_hidden_channels=config['kpnet_hidden_channels'],
                 kpnet_conv_size=config['kpnet_conv_size'],
                 dropout=config['dropout'],
                 diffusion_step_embed_dim_in=config['diffusion_step_embed_dim_in'],
                 diffusion_step_embed_dim_mid=config['diffusion_step_embed_dim_mid'],
                 diffusion_step_embed_dim_out=config['diffusion_step_embed_dim_out'],
                 use_weight_norm=config['use_weight_norm'])
        load_ckpt(vocoder, base_dir, 'model')

        # Init hyperparameters by linear schedule
        noise_schedule = torch.linspace(float(config["beta_0"]), float(config["beta_T"]), int(config["T"])).cuda()
        diffusion_hyperparams = compute_hyperparams_given_schedule(noise_schedule)

        # map diffusion hyperparameters to gpu
        for key in diffusion_hyperparams:
            if key in ["beta", "alpha", "sigma"]:
                diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
        diffusion_hyperparams = diffusion_hyperparams

        if config['noise_schedule'] != '':
            noise_schedule = config['noise_schedule']
            if isinstance(noise_schedule, list):
                noise_schedule = torch.FloatTensor(noise_schedule).cuda()
        else:
            # Select Schedule
            try:
                reverse_step = int(self.hparams.get('N'))
            except:
                print(
                    'Please specify $N (the number of revere iterations) in config file. Now denoise with 4 iterations.')
                reverse_step = 4
            if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000).cuda()
            elif reverse_step == 200:
                noise_schedule = torch.linspace(0.0001, 0.02, 200).cuda()

            # Below are schedules derived by Noise Predictor.
            # We will release codes of noise predictor training process & noise scheduling process soon. Please Stay Tuned!
            elif reverse_step == 8:
                noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                  0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593,
                                  0.5]
            elif reverse_step == 6:
                noise_schedule = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                  0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
            elif reverse_step == 4:
                noise_schedule = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
            elif reverse_step == 3:
                noise_schedule = [9.0000e-05, 9.0000e-03, 6.0000e-01]
            else:
                raise NotImplementedError

        if isinstance(noise_schedule, list):
            noise_schedule = torch.FloatTensor(noise_schedule).cuda()
        vocoder.eval()
        vocoder.to(self.device)
        return vocoder, diffusion_hyperparams, noise_schedule

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)
        audio_length = c.shape[-1] * self.hparams["hop_size"]
        y = sampling_given_noise_schedule(
            self.vocoder, (1, 1, audio_length), self.diffusion_hyperparams, self.noise_schedule, condition=c, ddim=False, return_sequence=False)
        return y

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

    @classmethod
    def example_run(cls, save_audio=True, return_audio=False):
        from utils.hparams import set_hparams
        from utils.hparams import hparams as hp
        from utils.audio import save_wav

        torch.manual_seed(time.time())

        set_hparams()
        infer_ins = cls(hp)
        with open('dictionaries/phone_uv_set.json', 'r', encoding='utf-8') as f:
            phone_uv_set = json.load(f)
            hp['phone_uv_set'] = set(phone_uv_set)
        with open(hp["proj"], 'r', encoding='utf-8') as f:
            project = json.load(f)
        result = []
        total_length = 0
        
        for i, segment in enumerate(project):
            out = infer_ins.infer_once(segment)
            os.makedirs('infer_out', exist_ok=True)
            offset = int(segment.get('offset', 0) * hp["audio_sample_rate"])
            out = np.concatenate([np.zeros(max(offset-total_length, 0)), out])
            total_length += len(out)
            result.append(out)
        
        audio = np.concatenate(result)
        if save_audio:
            save_wav(np.concatenate(result), f'infer_out/{hp["title"]}【{hp["exp_name"]}】.wav', hp['audio_sample_rate'])
        if return_audio:
            return audio