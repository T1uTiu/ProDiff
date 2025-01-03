import json
import librosa
import numpy as np
from scipy import interpolate
import torch
from torch.functional import F
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_spk_map
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur


@register_binarizer
class PitchPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.spk_map = build_spk_map(self.data_dir, self.datasets)
        # components
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        # param
        self.samplerate = hparams["audio_sample_rate"]
        self.hop_size, self.fft_size, self.win_size = hparams["hop_size"], hparams["fft_size"], hparams["win_size"]
        self.timesteps = self.hop_size / self.samplerate 
        self.midi_smooth = SinusoidalSmoothingConv1d(
            round(0.06 / self.timesteps)
        ).eval()
    
    @staticmethod
    def category():
        return "pitch"
    
    def load_meta_data(self):
        transcription_item_list = []
        for dataset in self.datasets:
            data_dir = dataset["data_dir"]
            spk_id = self.spk_map[dataset["speaker"]]
            with open(f"{data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            for label in labels:
                # note
                note_seq = label["note_seq"].split(" ")
                note_dur = [float(x) for x in label["note_dur"].split(" ")]
                item = {
                    "wav_fn" : f"{data_dir}/wav/{label['name']}.wav",
                    "spk_id" : spk_id,
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
        # spk
        preprocessed_item["spk_id"] = item["spk_id"]
        # f0
        f0, uv = self.pe.get_pitch(
            waveform, 
            samplerate = self.samplerate, 
            length = mel_len, 
            hop_size = self.hop_size, 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item['item_name']}, wav_fn: {item['wav_fn']}"
        preprocessed_item["pitch"] = librosa.hz_to_midi(f0.astype(np.float32))
        # note
        mel2note = get_mel2ph_dur(self.lr, torch.FloatTensor(item["note_dur"]), mel_len, self.timestep)
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
        # base f0
        frame_pitch = torch.gather(F.pad(torch.FloatTensor(note_midi), [1, 0], value=-1), 0, torch.LongTensor(mel2note))
        preprocessed_item["base_pitch"] = self.midi_smooth(frame_pitch[None])[0].detach().numpy()
        return preprocessed_item