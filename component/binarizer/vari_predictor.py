import json
import librosa
import numpy as np
from scipy import interpolate
import torch
from torch.functional import F
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_lang_map, build_phone_encoder, build_spk_map, extract_harmonic_aperiodic, get_breath, get_energy, get_tension, get_voicing
from component.pe.base import get_pitch_extractor_cls
from modules.commons.common_layers import SinusoidalSmoothingConv1d
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur

@register_binarizer
class VariPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        binarization_args = hparams["binarization_args"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # basic info
        self.ph_map, self.ph_encoder = build_phone_encoder(self.data_dir, hparams["dictionary"])
        self.need_spk_id = binarization_args.get("with_spk_id", True)
        if self.need_spk_id:
            self.spk_map = build_spk_map(self.data_dir, self.datasets)
        self.need_lang_id = binarization_args.get("with_lang_id", True)
        if self.need_lang_id:
            self.lang_map = build_lang_map(self.data_dir, hparams["dictionary"])
        # components
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        # param
        self.samplerate = hparams["audio_sample_rate"]
        self.hop_size, self.fft_size, self.win_size = hparams["hop_size"], hparams["fft_size"], hparams["win_size"]
        self.timesteps = self.hop_size / self.samplerate 
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
    
    @staticmethod
    def category():
        return "vari"

    def load_meta_data(self):
        transcription_item_list = []
        for dataset in self.datasets:
            data_dir = dataset["data_dir"]
            spk_id = self.spk_map[dataset["speaker"]]
            lang = dataset["language"]
            lang_id = self.lang_map[lang]
            with open(f"{data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            for item_name, label in labels.items():
                # ph
                ph_text = [f"{x}/{lang}" for x in label["ph_seq"].split(" ")]
                ph_dur = [float(x) for x in label["ph_dur"].split(" ")]
                ph_seq = self.ph_encoder.encode(ph_text)
                # note
                note_seq = label["note_seq"].split(" ")
                note_dur = [float(x) for x in label["note_dur"].split(" ")]
                item = {
                    "wav_fn" : f"{data_dir}/wav/{item_name}.wav",
                    "ph_seq" : ph_seq,
                    "ph_dur" : ph_dur,
                    "note_seq": note_seq,
                    "note_dur": note_dur,
                }
                if self.need_spk_id:
                    item["spk_id"] = spk_id
                if self.need_lang_id:
                    item["lang_seq"] = [lang_id]*len(ph_seq)
                transcription_item_list.append(item)
        return transcription_item_list

    def process_item(self, item: dict):
        hparams = self.hparams
        preprocessed_item = {
            "ph_seq" : np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur" : np.array(item["ph_dur"], dtype=np.float32),
        }
        # wavform
        waveform, _ = librosa.load(item["wav_fn"], sr=self.samplerate)
        mel_len = round(len(waveform) / self.hop_size)
        # spk
        if self.need_spk_id:
            preprocessed_item["spk_id"] = item["spk_id"]
        if self.need_lang_id:
            preprocessed_item["lang_seq"] = np.array(item["lang_seq"], dtype=np.int64)
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
        # ph dur
        preprocessed_item["mel2ph"] = get_mel2ph_dur(self.lr, torch.FloatTensor(item["ph_dur"]), mel_len, self.timesteps)
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
        if self.need_voicing:
            preprocessed_item["voicing"] = get_voicing(
                sp=harmonic_part,
                mel_len=mel_len,
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
                mel_len=mel_len,
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
                mel_len=mel_len,
                f0=f0,
                hop_size=self.hop_size,
                win_size=self.win_size,
                samplerate=self.samplerate,
                smooth_func=self.tension_smooth,
                device=self.device
            )
        return preprocessed_item
    