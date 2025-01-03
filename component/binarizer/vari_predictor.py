import json
import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import textgrid
import torch
from torch.functional import F
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import extract_harmonic_aperiodic, get_energy, get_mel_spec
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur
from component.vocoder.base_vocoder import get_vocoder_cls


@register_binarizer
class VariPredictorBinarizer(Binarizer):
    def __init__(self, hparams, vari_type):
        super().__init__(hparams)
        self.vari_type = vari_type
        # components
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        # param
        self.samplerate = hparams["audio_sample_rate"]
        self.hop_size, self.fft_size, self.win_size = hparams["hop_size"], hparams["fft_size"], hparams["win_size"]
        self.timesteps = self.hop_size / self.samplerate 
        # variance
        self.vari_smooth = SinusoidalSmoothingConv1d(
            round(0.12 / self.timesteps)
        ).eval().to(self.device)
    
    def load_meta_data(self):
        transcription_item_list = []
        for dataset in self.datasets:
            data_dir = dataset["data_dir"]
            with open(f"{data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            for label in labels:
                # note
                note_seq = label["note_seq"].split(" ")
                note_dur = [float(x) for x in label["note_dur"].split(" ")]
                item = {
                    "wav_fn" : f"{data_dir}/wav/{label['name']}.wav",
                    "note_seq": note_seq,
                    "note_dur": note_dur,
                }
                transcription_item_list.append(item)
        return transcription_item_list

    def process_item(self, item: dict):
        hparams = self.hparams
        preprocessed_item = {}
        # wavform
        waveform, _ = librosa.load(item["wav_fn"], sr=self.samplerate)
        mel_len = round(len(waveform) / self.hop_size)
        # summary
        preprocessed_item["sec"] = len(waveform) / self.samplerate
        preprocessed_item["length"] = mel_len
        # f0
        f0, uv = self.pe.get_pitch(
            waveform, 
            samplerate = self.samplerate, 
            length = mel_len, 
            hop_size = self.hop_size, 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item['item_name']}, wav_fn: {item['wav_fn']}"
        preprocessed_item["f0"] = f0
        # note
        mel2note = get_mel2ph_dur(self.lr, torch.FloatTensor(item["note_dur"]), mel_len, self.timesteps)
        preprocessed_item["mel2note"] = mel2note
        note_midi = np.array(
            [librosa.note_to_midi(nt, round_midi=False) if nt != "rest" else -1 for nt in item["note_seq"]],
        )
        note_rest = note_midi == -1
        interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
        note_midi[note_rest] = interp_func(np.where(note_rest)[0])
        preprocessed_item["note_midi"] = note_midi
        preprocessed_item["note_rest"] = note_rest
        # harmonic-noise separation
        harmonic_part, aperiodic_part = extract_harmonic_aperiodic(waveform, hparams["vr_ckpt"])
        # voicing
        if self.vari_type == "voicing":
            voicing = get_energy(harmonic_part, mel_len, self.hop_size, self.win_size)
            voicing = self.vari_smooth(torch.from_numpy(voicing).to(self.device)[None])[0]
            preprocessed_item["voicing"] = voicing.detach().cpu().numpy()
        # breath
        if self.vari_type == "breath":
            breath = get_energy(aperiodic_part, mel_len, self.hop_size, self.win_size)
            breath = self.vari_smooth(torch.from_numpy(breath).to(self.device)[None])[0]
            preprocessed_item["breath"] = breath.detach().cpu().numpy()
        return preprocessed_item
    

class VoicingPredictorBinarizer(VariPredictorBinarizer):
    def __init__(self, hparams):
        super().__init__(hparams, "voicing")
    
    @staticmethod
    def category():
        return "voicing"
    
class BreathPredictorBinarizer(VariPredictorBinarizer):
    def __init__(self, hparams):
        super().__init__(hparams, "breath")
    
    @staticmethod
    def category():
        return "breath"