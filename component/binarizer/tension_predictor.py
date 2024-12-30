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
from component.binarizer.binarizer_utils import build_spk_map
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur
from vocoders.base_vocoder import get_vocoder_cls


@register_binarizer
class TensionPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        self.vocoder = get_vocoder_cls(hparams["vocoder"])()
        timesteps = hparams["hop_size"] / hparams["audio_sample_rate"]
        self.midi_smooth = SinusoidalSmoothingConv1d(
            round(0.06 / timesteps)
        ).eval()
    
    @staticmethod
    def category():
        return "pitch"
    
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
        lr, pe = self.lr, self.pe

        wav, mel = self.vocoder.wav2spec(item["wav_fn"], hparams=hparams)
        preprocessed_item = {}
        preprocessed_item["sec"] = len(wav) / hparams['audio_sample_rate']
        preprocessed_item["length"] = mel.shape[0]
        # f0
        f0, uv = pe.get_pitch(
            wav, 
            samplerate = hparams['audio_sample_rate'], 
            length = mel.shape[0], 
            hop_size = hparams['hop_size'], 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item['item_name']}, wav_fn: {item['wav_fn']}"
        preprocessed_item["f0"] = f0
        preprocessed_item["pitch"] = librosa.hz_to_midi(f0.astype(np.float32))
        # note
        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        mel2note = get_mel2ph_dur(lr, torch.FloatTensor(item["note_dur"]), mel.shape[0], timestep)
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
        # tension
        return preprocessed_item