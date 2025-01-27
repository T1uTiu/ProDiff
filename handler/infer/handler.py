import json
import os
import time
from typing import List
import librosa
from scipy import interpolate
from torch.functional import F
import numpy as np
import torch

from component.binarizer.binarizer_utils import extract_harmonic_aperiodic, get_kth_harmonic
from component.inferer.base import get_inferer_cls
from modules.svs.prodiff_teacher import ProDiffTeacher
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.audio import cross_fade, save_wav
from utils.ckpt_utils import load_ckpt
from utils.data_gen_utils import get_mel2ph_dur
from utils.pitch_utils import resample_align_curve, shift_pitch
from utils.text_encoder import TokenTextEncoder
from utils.hparams_v2 import set_hparams
from component.vocoder.base_vocoder import get_vocoder_cls


class InferHandler:
    def __init__(self, exp_name, 
                 pred_dur=False, pred_pitch=False, pred_voicing=False, pred_breath=False,
                 isolate_aspiration=False, isolate_base_harmonic=False):
        self.hparams = set_hparams(
            exp_name=exp_name,
            task="svs",
            make_work_dir=False
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.work_dir = self.hparams['work_dir']

        self.hop_size = self.hparams['hop_size']
        self.win_size = self.hparams['win_size']
        self.audio_sample_rate = self.hparams['audio_sample_rate']
        self.timestep = self.hop_size / self.audio_sample_rate

        self.ph_map, self.ph_encoder = self.build_phone_encoder()
        self.spk_map = self.build_spk_map()
        self.lang_map = self.build_lang_map()

        self.lr = LengthRegulator()
        self.build_model()

        self.pred_dur = pred_dur
        if pred_dur:
            is_local_pred_dur_model = os.path.exists(f"{self.work_dir}/dur/config.yaml")
            dur_pred_hparams = set_hparams(
                exp_name=exp_name if is_local_pred_dur_model else None,
                task="dur",
                global_hparams=False
            )
            self.dur_predictor = get_inferer_cls("dur")(dur_pred_hparams)
            self.dur_predictor.build_model(self.ph_encoder)
        self.pred_pitch = pred_pitch != ""
        if self.pred_pitch:
            self.pred_pitch_spk_id = self.spk_map[pred_pitch]
            is_local_pred_pitch_model = os.path.exists(f"{self.work_dir}/pitch/config.yaml")
            pitch_pred_hparams = set_hparams(
                exp_name=exp_name if is_local_pred_pitch_model else None,
                task="pitch",
                global_hparams=False,
                make_work_dir=False
            )
            self.pitch_predictor = get_inferer_cls("pitch")(pitch_pred_hparams)
            self.pitch_predictor.build_model()
            self.midi_smooth = SinusoidalSmoothingConv1d(round(0.06/self.timestep)).eval()
        self.pred_voicing = pred_voicing
        if self.pred_voicing:
            voicing_work_dir = os.path.join(*self.work_dir.split('\\')[:-1], 'voicing')
            is_local_pred_voicing_model = os.path.exists(f"{voicing_work_dir}/config.yaml")
            voicing_pred_hparams = set_hparams(
                exp_name=exp_name if is_local_pred_voicing_model else None,
                task="voicing",
                global_hparams=False,
                make_work_dir=False
            )
            self.voicing_predictor = get_inferer_cls("voicing")(voicing_pred_hparams)
            self.voicing_predictor.build_model()
        self.pred_breath = pred_breath
        if self.pred_breath:
            breath_work_dir = os.path.join(*self.work_dir.split('\\')[:-1], 'breath')
            is_local_pred_breath_model = os.path.exists(f"{breath_work_dir}/config.yaml")
            breath_pred_hparams = set_hparams(
                exp_name=exp_name if is_local_pred_breath_model else None,
                task="breath",
                global_hparams=False,
                make_work_dir=False
            )
            self.breath_predictor = get_inferer_cls("breath")(breath_pred_hparams)
            self.breath_predictor.build_model()
        self.vocoder = self.build_vocoder()
        self.isolate_aspiration = isolate_aspiration
        self.isolate_base_harmonic = isolate_base_harmonic
    
    def build_phone_encoder(self):
        ph_map_fn = os.path.join(self.work_dir, 'phone_set.json')
        with open(ph_map_fn, 'r') as f:
            ph_map = json.load(f)
        ph_list = list(sorted(set(ph_map.values())))
        return ph_map, TokenTextEncoder(None, vocab_list=ph_list, replace_oov='SP')

    def build_spk_map(self):
        spk_map_fn = os.path.join(self.work_dir, 'spk_map.json')
        assert os.path.exists(spk_map_fn), f"Speaker map file {spk_map_fn} not found"
        with open(spk_map_fn, 'r') as f:
            spk_map = json.load(f)
        return spk_map
    
    def build_lang_map(self):
        lang_map_fn = os.path.join(self.work_dir, 'lang_map.json')
        assert os.path.exists(lang_map_fn), f"Language map file {lang_map_fn} not found"
        with open(lang_map_fn, 'r') as f:
            lang_map = json.load(f)
        return lang_map

    def build_model(self):
        f0_stats_fn = f'{self.hparams["work_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            self.hparams['f0_mean'], self.hparams['f0_std'] = np.load(f0_stats_fn)
            self.hparams['f0_mean'] = float(self.hparams['f0_mean'])
            self.hparams['f0_std'] = float(self.hparams['f0_std'])
        model = ProDiffTeacher(len(self.ph_encoder), self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model', strict=False)
        model.to(self.device)
        self.model = model

    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        f0_seq = inp['f0_seq']
        mel2ph = inp['mel2ph']
        spk_mix_embed = inp.get('spk_mix_embed', None)
        gender_mix_embed = inp.get("gender_mix_embed", None)
        lang_seq = inp.get('lang_seq', None)
        vociing = inp.get('voicing', None)
        breath = inp.get('breath', None)
        mel_out = self.model(
            ph_seq, f0=f0_seq, mel2ph=mel2ph, 
            spk_mix_embed=spk_mix_embed, gender_mix_embed=gender_mix_embed,
            lang_seq=lang_seq, 
            voicing=vociing, breath=breath,
            infer=True
        )
        return mel_out

    def build_vocoder(self):
        vocoder = get_vocoder_cls(self.hparams["vocoder"])(self.hparams)
        vocoder.to_device(self.device)
        return vocoder

    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def get_speaker_mix(self, spk_name):
        if spk_name is None or spk_name == '':
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

    def get_gender_mix(self, gender_value:float):
        assert 0 <= gender_value <= 1, "gender must be in [0, 1]"
        gender_mix_id = torch.LongTensor([0, 1]).to(self.device)[None, None]
        gender_mix_value = torch.FloatTensor([1-gender_value, gender_value]).to(self.device)[None, None]
        return gender_mix_id, gender_mix_value

    def get_note_dur(self, note_dur, note_slur):
        note_num = len(note_dur)
        slow, fast = -1, 0
        while fast < note_num:
            if note_slur[fast] == 0:
                slow += 1
                note_dur[slow] = note_dur[fast]
            else:
                note_dur[slow] += note_dur[fast]
            fast += 1
        return note_dur[:slow+1]
    
    def get_ph_text(self, ph, lang=None):
        if not self.hparams["use_lang_id"]:
            return ph
        return f"{ph}/{lang}" if "/" not in ph else ph

    def infer(self, segment: dict):
        lang = segment.get("lang", None)
        ph_text_seq = [
            self.ph_map[self.get_ph_text(ph, lang)] for ph in segment["ph_seq"].split()
        ]

        if self.hparams["use_lang_id"]:
            lang_seq = torch.LongTensor(
                [self.lang_map[lang]] * len(ph_text_seq)
            ).to(self.device)[None, :] # [B=1, T_txt]

        ph_token_seq = torch.LongTensor(
            self.ph_encoder.encode(ph_text_seq)
        ).to(self.device)[None, :] # [B=1, T_txt]
        # dur
        if self.pred_dur:
            ph_num = torch.LongTensor([int(num) for num in segment["ph_num"].split()])
            ph2word = self.lr(ph_num[None])[0]
            onset = torch.diff(ph2word, dim=0, prepend=ph2word.new_zeros(1)).to(self.device)[None, :]
            note_dur = self.get_note_dur(
                note_dur=[float(x) for x in segment["note_dur"].split()], 
                note_slur=[int(x) for x in segment["note_slur"].split()]
            )
            word_dur = torch.FloatTensor(note_dur)[None, :] # [B=1, T_w]
            word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 1, ph2word[None, :]).to(self.device)# [B=1, T_txt]
            ph_dur = self.dur_predictor.run_model(
                ph_seq=ph_token_seq,
                onset=onset,
                word_dur=word_dur,
                ph_num=ph_num,
                note_dur=note_dur
            ).to(self.device)
        else:
            ph_dur = torch.from_numpy(np.array(segment["ph_dur"].split(), np.float32)).to(self.device)
        # mel2ph
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, ph_token_seq == 0)  # => [B=1, T]
        mel_len = mel2ph.shape[1]
        # note
        if self.pred_breath or self.pred_pitch or self.pred_voicing:
            note_midi = np.array(
                [librosa.note_to_midi(nt, round_midi=False) if nt != "rest" else -1 for nt in segment["note_seq"].split()],
                dtype=np.float32
            )
            note_rest = note_midi == -1
            if np.all(note_rest):
                note_midi = np.full_like(note_midi, 60.)
            else:
                interp_func = interpolate.interp1d(
                    np.where(~note_rest)[0], note_midi[~note_rest],
                    kind='nearest', fill_value='extrapolate'
                )
                note_midi[note_rest] = interp_func(np.where(note_rest)[0])
            note_rest = torch.BoolTensor(note_rest)
            note_midi = torch.FloatTensor(note_midi)
            note_dur_sec = torch.from_numpy(np.array(segment["note_dur_seq"].split(), np.float32))
            mel2note = torch.from_numpy(get_mel2ph_dur(self.lr, note_dur_sec, mel_len, self.timestep))
        # pitch
        if self.pred_pitch:
            expr = segment.get("pitch_expr", 1.)
            pitch_expr = torch.FloatTensor([expr]).to(self.device)[None, :]
            spk_id = torch.LongTensor([self.pred_pitch_spk_id]).to(self.device)
            base_f0 = torch.gather(F.pad(note_midi, [1, 0], value=-1), 0, mel2note)
            base_f0 = self.midi_smooth(base_f0[None])[0].to(self.device)[None, :]
            f0_seq = self.pitch_predictor.run_model(
                note_midi = note_midi.to(self.device)[None, :],
                note_rest = note_rest.to(self.device)[None, :],
                mel2note = mel2note.to(self.device)[None, :],
                base_f0 = base_f0,
                pitch_expr = pitch_expr,
                spk_id = spk_id,
            )
            f0_seq += base_f0
            f0_seq = f0_seq[0].cpu().detach().numpy()
            f0_seq = librosa.midi_to_hz(f0_seq)
        else:
            f0_seq = resample_align_curve(
                np.array(segment['f0_seq'].split(), np.float32),
                original_timestep=float(segment['f0_timestep']),
                target_timestep=self.timestep,
                align_length=mel2ph.shape[1]
            )
        f0_seq = torch.from_numpy(f0_seq)[None, :].to(self.device) # [B=1, T_mel]

        keyshift = segment.get("keyshift", 0)
        if keyshift != 0:
            f0_seq = shift_pitch(f0_seq, keyshift)
        
        spk_mix_embed = None
        if self.hparams["use_spk_id"]:
            spk_mix_id, spk_mix_value = self.get_speaker_mix(segment["spk_name"])
            spk_mix_embed = torch.sum(
                self.model.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3), 
                dim=2, keepdim=False
            )
        gender_mix_embed = None
        if self.hparams["use_gender_id"]:
            gender_value = segment["gender"]
            gender_mix_id, gender_mix_value = self.get_gender_mix(gender_value)
            gender_mix_embed = torch.sum(
                self.model.gender_embed(gender_mix_id) * gender_mix_value.unsqueeze(3),
                dim=2, keepdim=False
            )
        voicing = None
        if self.hparams.get("use_voicing_embed", False):
            if "voicing" in segment:
                voicing = torch.FloatTensor([float(x) for x in segment["voicing"].split()]).to(self.device)[None, :]
            else:
                if self.pred_voicing:
                    voicing = self.voicing_predictor.run_model(
                        note_midi = note_midi.to(self.device)[None, :],
                        note_rest = note_rest.to(self.device)[None, :],
                        mel2note = mel2note.to(self.device)[None, :],
                        f0 = f0_seq,
                    )
                else:
                    voicing = torch.FloatTensor([-10.0] * mel_len).to(self.device)[None, :]
        breath = None
        if self.hparams.get("use_breath_embed", False):
            if "breath" in segment:
                breath = torch.FloatTensor([float(x) for x in segment["breath"].split()]).to(self.device)[None, :]
            else:
                if self.pred_breath:
                    breath = self.breath_predictor.run_model(
                        note_midi = note_midi.to(self.device)[None, :],
                        note_rest = note_rest.to(self.device)[None, :],
                        mel2note = mel2note.to(self.device)[None, :],
                        f0 = f0_seq,
                    )
                else:
                    breath = torch.FloatTensor([-50.0] * mel_len).to(self.device)[None, :]
        with torch.no_grad():
            start_time = time.time()
            mel_out = self.run_model(
                ph_seq=ph_token_seq, 
                f0_seq=f0_seq, 
                mel2ph=mel2ph, 
                spk_mix_embed=spk_mix_embed, 
                gender_mix_embed=gender_mix_embed,
                lang_seq=lang_seq, 
                voicing=voicing,
                breath=breath,
                infer=True
            )
            print(f"Inference Time: {time.time() - start_time}")
            wav_out = self.run_vocoder(mel_out, f0=f0_seq)
            
        wav_out = wav_out.squeeze().cpu().numpy()
        if self.isolate_aspiration:
            sp, ap = extract_harmonic_aperiodic(wav_out, self.hparams["vr_ckpt"])
            if self.isolate_base_harmonic:
                f0 = f0_seq.squeeze().cpu().numpy()
                base_harmonic = get_kth_harmonic(0, sp, f0, self.hop_size, self.win_size, self.audio_sample_rate)
                return sp-base_harmonic, ap, base_harmonic
            return sp, ap
        return [wav_out]
        

    def handle(self, proj: List[dict] = None, proj_fn=None, spk_name=None, lang=None, keyshift=0, gender=0):
        if proj is None:
            with open(proj_fn, 'r', encoding='utf-8') as f:
                proj = json.load(f)
        result, total_length = [np.zeros(0)], [0]
        if self.isolate_aspiration:
            result.append(np.zeros(0))
            total_length.append(0)
        if self.isolate_base_harmonic:
            result.append(np.zeros(0))
            total_length.append(0)
        for segment in proj:
            segment.setdefault('lang', lang)
            segment.setdefault("keyshift", int(keyshift))
            segment.setdefault('spk_name', spk_name)
            segment["gender"]= float(gender)
            out = self.infer(segment)
            offset = [round(segment.get('offset', 0) * self.audio_sample_rate) - total_length[i] for i in range(len(result))]
            for i, (offset_part, out_part) in enumerate(zip(offset, out)):
                if offset_part >= 0:
                    result[i] = np.append(result[i], np.zeros(offset_part))
                    result[i] = np.append(result[i], out_part)
                else:
                    result[i] = cross_fade(result[i], out_part, total_length[i] + offset_part)
            for i in range(len(result)):
                total_length[i] += offset[i]
                total_length[i] += out[i].shape[0]
        title = proj_fn.split('/')[-1].split('.')[0]
        if not self.isolate_aspiration:
            out_fn = f'infer_out/{title}【{self.hparams["exp_name"]}】.wav'
            save_wav(result[0], out_fn, self.audio_sample_rate)
        else:
            out_fn_sp = f'infer_out/{title}_sp【{self.hparams["exp_name"]}】.wav'
            save_wav(result[0], out_fn_sp, self.audio_sample_rate)
            out_fn_ap = f'infer_out/{title}_ap【{self.hparams["exp_name"]}】.wav'
            save_wav(result[1], out_fn_ap, self.audio_sample_rate)
            if self.isolate_base_harmonic:
                out_fn_bh = f'infer_out/{title}_bh【{self.hparams["exp_name"]}】.wav'
                save_wav(result[2], out_fn_bh, self.audio_sample_rate)