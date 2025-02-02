import json
import random

import librosa
import numpy as np
import torch
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_lang_map, build_phone_encoder, build_spk_map, extract_harmonic_aperiodic, get_breath, get_energy, get_mel_spec, get_tension, get_voicing
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from modules.svs.prodiff_teacher import ProDiffTeacher
from utils.ckpt_utils import load_ckpt
from utils.data_gen_utils import get_mel2ph_dur

@register_binarizer
class SVSBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        binarization_args = hparams["binarization_args"]
        # basic info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ph_map, self.ph_encoder = build_phone_encoder(self.data_dir, hparams["dictionary"], hparams["languages"])

        self.need_spk_id = binarization_args.get("with_spk_id", True)
        if self.need_spk_id:
            self.spk_map = build_spk_map(self.data_dir, self.datasets)

        self.need_lang_id = binarization_args.get("with_lang_id", True)
        if self.need_lang_id:
            self.lang_map = build_lang_map(self.data_dir, hparams["languages"])
        
        # param
        self.samplerate = hparams["audio_sample_rate"]
        self.hop_size, self.fft_size, self.win_size = hparams["hop_size"], hparams["fft_size"], hparams["win_size"]
        self.timesteps = self.hop_size / self.samplerate
        self.f_min, self.f_max = hparams["fmin"], hparams["fmax"]   
        self.num_mel_bins = hparams["audio_num_mel_bins"]

        # components
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)

        # variance
        self.need_voicing = binarization_args.get("with_voicing", False)
        if self.need_voicing:
            self.voicing_smooth = SinusoidalSmoothingConv1d(
                round(0.12 / self.timesteps)
            ).eval().to(self.device)

        self.need_breath = binarization_args.get("with_breath", False)
        if self.need_breath:
            self.breath_smooth = SinusoidalSmoothingConv1d(
                round(0.12 / self.timesteps)
            ).eval().to(self.device)

        self.need_tension = binarization_args.get("with_tension", False)
        if self.need_tension:
            self.tension_smooth = SinusoidalSmoothingConv1d(
                round(0.12 / self.timesteps)
            ).eval().to(self.device)

        # post process
        if binarization_args['shuffle']:
            random.seed(3407)
            random.shuffle(self.transcription_item_list)

    @staticmethod
    def category():
        return "svs"

    def load_meta_data(self):
        transcription_item_list = []
        for dataset in self.datasets:
            data_dir = dataset["data_dir"]
            lang = dataset["language"]
            lang_id = self.lang_map[lang]
            spk_id = self.spk_map[dataset["speaker"]]
            with open(f"{data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            for item_name, label in labels.items():
                ph_text = [self.ph_map[f"{x}/{lang}"] for x in label["ph_seq"].split(" ")]
                ph_dur = [float(x) for x in label["ph_dur"].split(" ")]
                ph_seq = self.ph_encoder.encode(ph_text)
                item = {
                    "wav_fn" : f"{data_dir}/wav/{item_name}.wav",
                    "ph_seq" : ph_seq,
                    "ph_dur" : ph_dur,
                }
                if self.need_spk_id:
                    item["spk_id"] = spk_id
                if self.need_lang_id:
                    item["lang_seq"] = [lang_id]*len(ph_seq)
                if self.hparams["use_gender_id"]:
                    item["gender_id"] = dataset["gender"]
                transcription_item_list.append(item)
        return transcription_item_list

    def process_item(self, item: dict):
        hparams = self.hparams
        preprocessed_item = {
            "ph_seq" : np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur" : np.array(item["ph_dur"], dtype=np.float32),
        }
        if self.need_spk_id:
            preprocessed_item["spk_id"] = item["spk_id"]
        if self.need_lang_id:
            preprocessed_item["lang_seq"] = np.array(item["lang_seq"], dtype=np.int64)
        # wavform
        waveform, _ = librosa.load(item["wav_fn"], sr=self.samplerate)
        # harmonic-aperiodic separation
        if self.need_voicing or self.need_breath or self.need_tension:
            harmonic_part, aperiodic_part = extract_harmonic_aperiodic(waveform, hparams["vr_ckpt"])
        # mel
        mel = get_mel_spec(
            waveform, 
            self.samplerate, self.num_mel_bins, 
            self.fft_size, self.win_size, self.hop_size, 
            self.f_min, self.f_max
        )
        preprocessed_item["mel"] = mel
        # summary
        preprocessed_item["sec"] = len(waveform) / self.samplerate
        preprocessed_item["length"] = mel.shape[0]
        # gender
        if hparams["use_gender_id"]:
            preprocessed_item["gender_id"] = item["gender_id"],
        # dur
        preprocessed_item["mel2ph"] = get_mel2ph_dur(self.lr, torch.FloatTensor(item["ph_dur"]), mel.shape[0], self.timesteps)
        # f0
        f0, uv = self.pe.get_pitch(
            waveform, 
            samplerate = self.samplerate, 
            length = mel.shape[0], 
            hop_size = self.hop_size, 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item['item_name']}, wav_fn: {item['wav_fn']}"
        preprocessed_item["f0"] = f0
        # voicing
        if self.need_voicing:
            preprocessed_item["voicing"] = get_voicing(
                sp=harmonic_part,
                mel_len=mel.shape[0],
                hop_size=self.hop_size,
                win_size=self.win_size,
                smooth_func=self.voicing_smooth,
                norm=hparams["voicing_norm"],
                db_min=hparams["voicing_db_min"],
                db_max=hparams["voicing_db_max"],
                device=self.device
            )
        # breath
        if self.need_breath:
            preprocessed_item["breath"] = get_breath(
                ap=aperiodic_part,
                mel_len=mel.shape[0],
                hop_size=self.hop_size,
                win_size=self.win_size,
                smooth_func=self.breath_smooth,
                norm=hparams["breath_norm"],
                db_min=hparams["breath_db_min"],
                db_max=hparams["breath_db_max"],
                device=self.device
            )
        # tension
        if self.need_tension:
            preprocessed_item["tension"] = get_tension(
                sp=harmonic_part,
                mel_len=mel.shape[0],
                f0=f0,
                hop_size=self.hop_size,
                win_size=self.win_size,
                samplerate=self.samplerate,
                smooth_func=self.tension_smooth,
                device=self.device
            )
        return preprocessed_item

class SVSRectifiedDiffusionBinarizer(SVSBinarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        teacher_ckpt = hparams["teacher_ckpt"]
        self.teacher = ProDiffTeacher(len(self.ph_encoder), hparams)
        load_ckpt(self.teacher, teacher_ckpt, "model")
        self.teacher.eval()
        self.teacher.to(self.device)

    @staticmethod
    def category():
        return "svs_rectified"
    
    def process_item(self, item: dict):
        preprocessed_item = super().process_item(item)
        ph_seq = torch.LongTensor(preprocessed_item["ph_seq"]).to(self.device)[None, :]
        mel2ph = torch.LongTensor(preprocessed_item["mel2ph"]).to(self.device)[None, :]
        f0 = torch.FloatTensor(preprocessed_item["f0"]).to(self.device)[None, :]
        if self.hparams["use_spk_id"]:
            spk_id = torch.LongTensor([preprocessed_item["spk_id"]]).to(self.device)
        if self.hparams["use_gender_id"]:
            gender_id = torch.LongTensor([preprocessed_item["gender_id"]]).to(self.device)
        if self.hparams["use_lang_id"]:
            lang_seq = torch.LongTensor(preprocessed_item["lang_seq"]).to(self.device)[None, :]
        if self.hparams["use_voicing_embed"]:
            voicing = torch.FloatTensor(preprocessed_item["voicing"]).to(self.device)[None, :]
        if self.hparams["use_breath_embed"]:
            breath = torch.FloatTensor(preprocessed_item["breath"]).to(self.device)[None, :]

        with torch.no_grad():
            condition = self.teacher.forward_condition(
                ph_seq, mel2ph, f0,
                lang_seq=lang_seq,
                spk_embed_id=spk_id, gender_embed_id=gender_id,
                voicing=voicing, breath=breath
            )
            b, T, device = condition.shape[0], condition.shape[1], condition.device
            x_T = torch.randn(b, 1, self.num_mel_bins, T, device=device)
            x_0 = self.teacher.diffusion(condition, x_T, infer=True)
            x_T = x_T.squeeze(1).transpose(-2, -1)
            preprocessed_item["condition"] = condition.squeeze(1).detach().cpu().numpy() # [T, Hidden]
            preprocessed_item["x_T"] = x_T.squeeze(1).detach().cpu().numpy() # [T, M]
            preprocessed_item["x_0"] = x_0.detach().cpu().numpy() # [T, M]
        return preprocessed_item