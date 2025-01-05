import json
import random

import librosa
import numpy as np
import torch
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_lang_map, build_phone_encoder, build_spk_map, extract_harmonic_aperiodic, get_breath, get_energy, get_mel_spec, get_voicing
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur

@register_binarizer
class SVSBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        # basic info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ph_map, self.ph_encoder = build_phone_encoder(self.data_dir, hparams["dictionary"])
        self.lang_map = build_lang_map(self.data_dir, hparams["dictionary"])
        self.spk_map = build_spk_map(self.data_dir, self.datasets)
        # param
        self.samplerate = hparams["audio_sample_rate"]
        self.hop_size, self.fft_size, self.win_size = hparams["hop_size"], hparams["fft_size"], hparams["win_size"]
        self.timesteps = self.hop_size / self.samplerate
        self.f_min, self.f_max = hparams["fmin"], hparams["fmax"]   
        self.num_mel_bins = hparams["audio_num_mel_bins"]
        self.ha_sep = self.hparams.get("use_voicing_embed", False) or self.hparams.get("use_breath_embed", False)
        # components
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        # variance
        if hparams.get("use_voicing_embed", False):
            self.voicing_smooth = SinusoidalSmoothingConv1d(
                round(0.12 / self.timesteps)
            ).eval().to(self.device)
        if hparams.get("use_breath_embed", False):
            self.breath_smooth = SinusoidalSmoothingConv1d(
                round(0.12 / self.timesteps)
            ).eval().to(self.device)
        # post process
        binarization_args = hparams["binarization_args"]
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
                ph_text = [f"{x}/{lang}" for x in label["ph_seq"].split(" ")]
                ph_dur = [float(x) for x in label["ph_dur"].split(" ")]
                ph_seq = self.ph_encoder.encode(ph_text)
                item = {
                    "wav_fn" : f"{data_dir}/wav/{item_name}.wav",
                    "ph_seq" : ph_seq,
                    "ph_dur" : ph_dur,
                    "spk_id" : spk_id,
                    "lang_seq" : [lang_id]*len(ph_seq),
                }
                if self.hparams["use_gender_id"]:
                    item["gender_id"] = dataset["gender"]
                transcription_item_list.append(item)
        return transcription_item_list

    def process_item(self, item: dict):
        hparams = self.hparams
        preprocessed_item = {
            "spk_id" : item["spk_id"],
            "ph_seq" : np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur" : np.array(item["ph_dur"], dtype=np.float32),
            "lang_seq" : np.array(item["lang_seq"], dtype=np.int64),
        }
        # wavform
        waveform, _ = librosa.load(item["wav_fn"], sr=self.samplerate)
        # harmonic-aperiodic separation
        if self.ha_sep:
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
        if self.hparams.get("use_voicing_embed", False):
            preprocessed_item["voicing"] = get_voicing(
                sp=harmonic_part,
                mel_len=mel.shape[0],
                hop_size=self.hop_size,
                win_size=self.win_size,
                smooth_moudle=self.voicing_smooth,
                norm=hparams["voicing_norm"],
                db_min=hparams["voicing_db_min"],
                db_max=hparams["voicing_db_max"],
                device=self.device
            )
        # breath
        if self.hparams.get("use_breath_embed", False):
            preprocessed_item["breath"] = get_breath(
                ap=aperiodic_part,
                mel_len=mel.shape[0],
                hop_size=self.hop_size,
                win_size=self.win_size,
                smooth_moudle=self.breath_smooth,
                norm=hparams["breath_norm"],
                db_min=hparams["breath_db_min"],
                db_max=hparams["breath_db_max"],
                device=self.device
            )
        return preprocessed_item