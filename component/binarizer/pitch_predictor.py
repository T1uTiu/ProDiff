import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import textgrid
import torch
from torch.functional import F
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_lang_map, build_phone_encoder, build_spk_map, SinusoidalSmoothingConv1d
from component.pe.base import get_pitch_extractor_cls
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur
from vocoders.base_vocoder import get_vocoder_cls


@register_binarizer
class PitchPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.spk_map = build_spk_map(self.data_dir, self.datasets)
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
            spk_id = self.spk_map[dataset["speaker"]]
            for tg_fn in os.listdir(f"{data_dir}/TextGrid"):
                if not tg_fn.endswith(".TextGrid"):
                    continue
                tg = textgrid.TextGrid.fromFile(f"{data_dir}/TextGrid/{tg_fn}")
                # note
                note_seq, note_dur = [], []
                for x in tg.getFirst("note"):
                    note_seq.append(x.mark)
                    note_dur.append(x.maxTime - x.minTime)
                item = {
                    "wav_fn" : f"{data_dir}/wav/{tg_fn.replace('.TextGrid', '.wav')}",
                    "spk_id" : spk_id,
                    "note_seq": note_seq,
                    "note_dur": note_dur,
                }
                transcription_item_list.append(item)
        return transcription_item_list

    def process_item(self, item: dict):
        hparams = self.hparams
        lr, pe = self.lr, self.pe

        wav, mel = self.vocoder.wav2spec(item["wav_fn"], hparams=hparams)
        preprocessed_item = {
            "spk_id" : item["spk_id"],
        }
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
        # base f0
        frame_pitch = torch.gather(F.pad(torch.FloatTensor(note_midi), [1, 0], value=-1), 0, torch.LongTensor(mel2note))
        preprocessed_item["base_pitch"] = self.midi_smooth(frame_pitch[None])[0].detach().numpy()
        preprocessed_item["base_f0"] = librosa.midi_to_hz(preprocessed_item["base_pitch"])
        return preprocessed_item