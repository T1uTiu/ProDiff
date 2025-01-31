import json
import os
import fastapi
import librosa
import numpy as np
from scipy import interpolate
import uvicorn
import torch
import torch.nn.functional as F
from itertools import chain
from typing import List

from component.binarizer.binarizer_utils import extract_harmonic_aperiodic
from component.inferer.base import get_inferer_cls
from component.vocoder.base_vocoder import get_vocoder_cls
from modules.svs.prodiff_teacher import ProDiffTeacher
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.ckpt_utils import load_ckpt
from utils.data_gen_utils import get_mel2ph_dur
from utils.hparams_v2 import set_hparams
from utils.pitch_utils import resample_align_curve
from utils.text_encoder import TokenTextEncoder
from . import config

class WebHandler:
    def __init__(self, exp_name):
        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_name = exp_name
        self.hparams = set_hparams(
            exp_name=exp_name,
            task="svs",
            make_work_dir=False
        )
        self.work_dir = self.hparams["work_dir"]
        self.spk_map = self.build_spk_map(os.path.join(self.work_dir, 'spk_map.json'))
        self.build_lang_map()
        self.build_phone_encoder()
        self.lr = LengthRegulator()
        self.timestep = self.hparams["hop_size"] / self.hparams["audio_sample_rate"]
        self.midi_smooth = SinusoidalSmoothingConv1d(round(0.06/self.timestep)).eval()
        self.build_model()
        self.build_vocoder()
        
        # dur predictor
        dur_predictor_work_dir = os.path.join(*self.work_dir.split('\\')[:-1], 'dur')
        is_local_pred_dur_model = os.path.exists(f"{dur_predictor_work_dir}/config.yaml")
        dur_pred_hparams = set_hparams(
            exp_name=exp_name if is_local_pred_dur_model else None,
            task="dur",
            global_hparams=False
        )
        self.dur_predictor = get_inferer_cls("dur")(dur_pred_hparams)
        self.dur_predictor.build_model(self.ph_encoder)
        # pitch predictor
        pitch_predictor_work_dir = os.path.join(*self.work_dir.split('\\')[:-1], 'pitch')
        is_local_pred_pitch_model = os.path.exists(f"{pitch_predictor_work_dir}/config.yaml")
        pitch_pred_hparams = set_hparams(
            exp_name=exp_name if is_local_pred_pitch_model else None,
            task="pitch",
            global_hparams=False,
            make_work_dir=False
        )
        self.build_ph_category_encoder(pitch_pred_hparams["work_dir"])
        self.pitch_pred_spk_map = self.build_spk_map(os.path.join(pitch_pred_hparams["work_dir"], 'spk_map.json'))
        self.pitch_predictor = get_inferer_cls("pitch")(pitch_pred_hparams)
        self.pitch_predictor.build_model(self.ph_category_encoder)
        self.midi_smooth = SinusoidalSmoothingConv1d(round(0.06/self.timestep)).eval()
        # web
        self.app = fastapi.FastAPI(debug=True)
        self.app.add_api_route(config.get_basic_info_api, self.api_get_basic_info, methods=['GET'])
        self.app.add_api_route(config.post_infer_api, self.api_infer, methods=["POST"])
        self.app.add_api_route(config.post_pred_dur_api, self.api_pred_dur, methods=["POST"])
        self.app.add_api_route(config.post_pred_pitch_api, self.api_pred_pitch, methods=["POST"])
    
    def build_spk_map(self, spk_map_fn):
        assert os.path.exists(spk_map_fn), f"Speaker map file {spk_map_fn} not found"
        with open(spk_map_fn, 'r') as f:
            spk_map = json.load(f)
        return spk_map
        
    def build_lang_map(self):
        lang_map_fn = os.path.join(self.work_dir, 'lang_map.json')
        assert os.path.exists(lang_map_fn), f"Language map file {lang_map_fn} not found"
        with open(lang_map_fn, 'r') as f:
            lang_map = json.load(f)
        self.lang_map = lang_map
    
    def build_ph_category_encoder(self, work_dir):
        # build ph_category_encoder
        ph_category_list_fn = os.path.join(work_dir, 'ph_category_list.json')
        with open(ph_category_list_fn, 'r') as f:
            ph_category_list: List[str] = json.load(f)
        self.ph_category_encoder = TokenTextEncoder(None, vocab_list=ph_category_list, replace_oov='SP')

    def build_phone_encoder(self):
        # build ph_map and ph_encoder
        ph_map_fn = os.path.join(self.work_dir, 'phone_set.json')
        with open(ph_map_fn, 'r') as f:
            ph_map = json.load(f)
        ph_list = list(sorted(set(ph_map.values())))
        self.ph_map = ph_map
        self.ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov='SP')
        # build dictionary
        self.word_dictionay = {}
        self.ph2category = {
            "AP": "AP",
            "SP": "SP"
        }
        self.consonant_set = {}
        for lang in self.hparams["languages"]:
            self.word_dictionay[lang] = {"AP": ["AP"], "SP": ["SP"]}
            with open(self.hparams["dictionary"][lang]["word"], 'r') as f:
                for x in f.readlines():
                    line = x.split("\n")[0].split('\t') # "zhi    zh ir"
                    word = line[0]
                    ph_list = line[1].split(' ')
                    self.word_dictionay[lang][word] = ph_list
            self.consonant_set[lang] = set()
            with open(self.hparams["dictionary"][lang]["phoneme"], "r") as f:
                for x in f.readlines():
                    line = x.split("\n")[0].split(' ') # "zh consonant affricate"
                    ph, ph_type, ph_category = line[0], line[1], line[2]
                    if ph_type == "consonant":
                        self.consonant_set[lang].add(ph)
                    self.ph2category[self.ph_map[self.get_ph_text(ph, lang)]] = ph_category
                    self.word_dictionay[lang][f".{ph}"] = [ph]


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
        self.vocoder = vocoder
    
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def get_speaker_mix(self, spk_name):
        if spk_name is None or spk_name == '':
            # Get the first speaker
            spk_mix_map = {self.spk_map.keys()[0]: 1.0}
        elif "|" in spk_name:
            spk_mix_map = dict([x.split(':') for x in spk_name.split('|')])
            for k in spk_mix_map:
                spk_mix_map[k] = float(spk_mix_map[k])
        else:
            spk_mix_map = {spk_name: 1.0}
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

    def get_ph_text(self, ph: str, lang=None):
        if not self.hparams["use_lang_id"]:
            return ph
        return f"{ph}/{lang}" if "/" not in ph else ph

    def get_ph_num_list(self, lang: str, word_ph_text_list: List[List[str]]) -> List[int]:
        num_word = len(word_ph_text_list)
        ph_num = [0] * num_word
        for i, ph_list in enumerate(word_ph_text_list):
            for ph_idx, ph in enumerate(ph_list):
                # if the first ph of the word is consonant, add it to the last word to align the beat
                if ph_idx == 0 and ph in self.consonant_set[lang]:
                    ph_num[i-1] += 1
                else:
                    ph_num[i] += 1
        return ph_num
    
    async def api_get_basic_info(self):
        return {
            "languages": list(self.lang_map.keys()),
            "speakers": list(self.spk_map.keys()),
            "hop_size": self.hparams["hop_size"],
            "samplerate": self.hparams["audio_sample_rate"],
            "pitch_styles": list(self.pitch_pred_spk_map.keys())
        }

    async def api_pred_pitch(self, req: dict):
        # check params
        assert "language" in req, "language is required"
        assert "ph_text_list" in req, "ph_text_list is required"
        assert "ph_dur_list" in req, "ph_dur_list is required"
        assert "note_midi_list" in req, "note_midi_list is required"
        assert "note_dur_list" in req, "note_dur_list is required"
        # process input
        lang = req["language"]
        ph_text_list = [self.ph2category[ph] for ph in req["ph_text_list"]]
        ph_token_seq = torch.LongTensor(
            self.ph_encoder.encode(ph_text_list)
        ).to(self.device)[None, :] # [B=1, T_txt]
        ph_dur_list = req["ph_dur_list"]
        ph_dur_seq = torch.FloatTensor(ph_dur_list).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_seq, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, ph_token_seq == 0)  # => [B=1, T]
        length = mel2ph.shape[1]
        note_midi = np.array(req["note_midi_list"], dtype=np.float32)
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
        note_dur_sec = torch.from_numpy(np.array(req["note_dur_list"], np.float32))
        mel2note = torch.from_numpy(get_mel2ph_dur(self.lr, note_dur_sec, length, self.timestep))
        expr = req.get("pitch_expr", 1.)
        pitch_expr = torch.FloatTensor([expr]).to(self.device)[None, :]
        speaker = req.get("style", "")
        spk_id = torch.LongTensor([self.spk_map.get(speaker, 0)]).to(self.device)
        base_pitch = torch.gather(F.pad(note_midi, [1, 0], value=-1), 0, mel2note)
        base_pitch = self.midi_smooth(base_pitch[None])[0].to(self.device)[None, :]
        # model infer
        delta_pitch = self.pitch_predictor.run_model(
            ph_seq = ph_token_seq,
            mel2ph = mel2ph,
            note_midi = note_midi.to(self.device)[None, :],
            note_rest = note_rest.to(self.device)[None, :],
            mel2note = mel2note.to(self.device)[None, :],
            base_pitch = base_pitch,
            pitch_expr = pitch_expr,
            spk_id = spk_id,
        )
        pitch = base_pitch + delta_pitch
        pitch = pitch[0].cpu().detach().numpy()
        return {
            "pitch": pitch.tolist()
        }
    
    async def api_pred_dur(self, req: dict):
        # check params
        assert "language" in req, "language is required"
        assert "word_list" in req, "word_list is required"
        assert "word_dur_list" in req, "word_dur_list is required"
        assert "start_time" in req, "start_time is required"

        # process input
        language = req["language"]

        word_list = ["SP"] + req["word_list"]
        word_ph_text_list = [ 
            self.word_dictionay[language].get(word, ["SP"]) for word in word_list
        ]
        ph_text_list = list(chain.from_iterable([
            [self.ph_map.get(self.get_ph_text(ph, language), "SP") for ph in ph_list] 
            for ph_list in word_ph_text_list
        ]))
        ph_token_seq = torch.LongTensor(
            self.ph_encoder.encode(ph_text_list)
        ).to(self.device)[None, :] # [B=1, T_txt]

        ph_num_list = self.get_ph_num_list(language, word_ph_text_list)
        ph_num = torch.LongTensor(ph_num_list)
        ph2word = self.lr(ph_num[None])[0]
        onset = torch.diff(ph2word, dim=0, prepend=ph2word.new_zeros(1)).to(self.device)[None, :]

        padding_note_time = req.get("padding_note_time", 0.5) # padding SP time, sec
        word_dur_list = [padding_note_time] + req["word_dur_list"]
        word_dur_seq = torch.FloatTensor(word_dur_list)[None, :] # [B=1, T_w]
        word_dur_seq = torch.gather(F.pad(word_dur_seq, [1, 0], value=0), 1, ph2word[None, :]).to(self.device)# [B=1, T_txt]

        # model infer
        ph_dur = self.dur_predictor.run_model(
            ph_seq=ph_token_seq,
            onset=onset,
            word_dur=word_dur_seq,
            ph_num=ph_num,
            note_dur=word_dur_list
        ).to(self.device)

        # post process
        segment_start_time = req["start_time"]
        start_time = segment_start_time - padding_note_time
        ph_dur_list: List[float] = ph_dur.cpu().detach().numpy().tolist()
        word_list = word_list[1:]
        note_ph_list = []
        idx = 0
        ph_start_time = start_time
        for i, word in enumerate(word_list):
            word_ph_num = len(self.word_dictionay[language].get(word, ["SP"]))
            # add padding SP to the first word
            if i == 0:
                word_ph_num += 1
            note_ph_list.append([])
            for i in range(idx, idx+word_ph_num):
                note_ph_list[-1].append({
                    "ph": ph_text_list[i],
                    "start_time": ph_start_time,
                    "end_time": ph_start_time + ph_dur_list[i]
                })
                ph_start_time += ph_dur_list[i]
            idx += word_ph_num
        return {
            "start_time": start_time,
            "note_ph_list": note_ph_list,
        }

    async def api_infer(self, req: dict):
        # check params
        assert "speaker" in req, "speaker is required"
        assert "language" in req, "language is required"
        assert "ph_text_list" in req, "ph_text_list is required"
        assert "ph_dur_list" in req, "ph_dur_list is required"
        assert "pitch_list" in req, "pitch_list is required"

        # speaker
        spk_mix_id, spk_mix_value = self.get_speaker_mix(req["speaker"])
        spk_mix_embed = torch.sum(
            self.model.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3), 
            dim=2, keepdim=False
        )
        # ph
        ph_text_list = req["ph_text_list"]
        ph_token_seq = torch.LongTensor(
            self.ph_encoder.encode(ph_text_list)
        ).to(self.device)[None, :] # [B=1, T_txt]
        # lang
        language = req["language"]
        lang_seq = torch.LongTensor(
            [self.lang_map[language]] * len(ph_text_list)
        ).to(self.device)[None, :] # [B=1, T_txt]
        

        ph_dur_list = req["ph_dur_list"]
        ph_dur_seq = torch.FloatTensor(ph_dur_list).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_seq, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, ph_token_seq == 0)  # => [B=1, T]
        mel_len = mel2ph.shape[1]

        # pitch
        f0_arr = librosa.midi_to_hz(np.array(req["pitch_list"]))
        f0 = torch.FloatTensor(f0_arr).to(self.device)
        if f0.shape[0] < mel_len:
            f0 = torch.cat((f0, torch.full((mel_len - f0.shape[0],), fill_value=f0[-1], device=self.device)), dim=0)
        elif f0.shape[0] > mel_len:
            f0 = f0[:mel_len]
        f0 = f0[None, :]

        with torch.no_grad():
            mel_out = self.run_model(
                ph_seq=ph_token_seq, 
                f0_seq=f0, 
                mel2ph=mel2ph, 
                spk_mix_embed=spk_mix_embed, 
                lang_seq=lang_seq, 
                infer=True
            )
            wav_out = self.run_vocoder(mel_out, f0=f0)
        wav_out = wav_out.squeeze().cpu().numpy()
        # 谐波分离
        sp, ap = extract_harmonic_aperiodic(wav_out, self.hparams["vr_ckpt"])
        voicing = resample_align_curve(
            np.array(req["voicing_list"]),
            original_timestep=self.timestep,
            target_timestep=1/self.hparams["audio_sample_rate"],
            align_length=wav_out.shape[0]
        )
        voicing = librosa.db_to_amplitude(voicing)
        sp *= voicing
        breath = resample_align_curve(
            np.array(req["breath_list"]),
            original_timestep=self.timestep,
            target_timestep=1/self.hparams["audio_sample_rate"],
            align_length=wav_out.shape[0]
        )
        breath = librosa.db_to_amplitude(breath)
        ap *= breath
        wav_out = sp + ap
        return {
            "wav": wav_out.tolist()
        }

    def handle(self):
        uvicorn.run(self.app, host=config.server_host, port=config.server_port)
