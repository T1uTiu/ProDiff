import json
import random

import numpy as np
import torch
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_lang_map, build_phone_encoder, build_spk_map
from component.pe.base import get_pitch_extractor_cls
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur
from component.vocoder.base_vocoder import get_vocoder_cls

@register_binarizer
class SVSBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.ph_map, self.ph_encoder = build_phone_encoder(self.data_dir, hparams["dictionary"])
        self.lang_map = build_lang_map(self.data_dir, hparams["dictionary"])
        self.spk_map = build_spk_map(self.data_dir, self.datasets)
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        self.vocoder = get_vocoder_cls(hparams["vocoder"])()
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
            for label in labels:
                ph_text = [f"{x}/{lang}" for x in label["ph_seq"].split(" ")]
                ph_dur = [float(x) for x in label["ph_dur"].split(" ")]
                ph_seq = self.ph_encoder.encode(ph_text)
                item = {
                    "wav_fn" : f"{data_dir}/wav/{label['name']}.wav",
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
        lr, pe = self.lr, self.pe

        wav, mel = self.vocoder.wav2spec(item["wav_fn"], hparams=hparams)
        preprocessed_item = {
            "mel" : mel,
            "spk_id" : item["spk_id"],
            "ph_seq" : np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur" : np.array(item["ph_dur"], dtype=np.float32),
            "lang_seq" : np.array(item["lang_seq"], dtype=np.int64),
        }
        if hparams["use_gender_id"]:
            preprocessed_item["gender_id"] = item["gender_id"],
        preprocessed_item["sec"] = len(wav) / hparams['audio_sample_rate']
        preprocessed_item["length"] = mel.shape[0]

        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        preprocessed_item["mel2ph"] = get_mel2ph_dur(lr, torch.FloatTensor(item["ph_dur"]), mel.shape[0], timestep)

        f0, uv = pe.get_pitch(
            wav, 
            samplerate = hparams['audio_sample_rate'], 
            length = mel.shape[0], 
            hop_size = hparams['hop_size'], 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item['item_name']}, wav_fn: {item['wav_fn']}"
        preprocessed_item["f0"] = f0

        return preprocessed_item