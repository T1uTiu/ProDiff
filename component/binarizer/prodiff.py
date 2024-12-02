import json
import os
import random

import numpy as np
import torch
from component.binarizer.base import Binarizer
from component.pe.base import get_pitch_extractor_cls
from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import get_mel2ph_dur
from utils.text_encoder import TokenTextEncoder
from vocoders.base_vocoder import get_vocoder_cls


class ProDiffBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.binary_data_dir = hparams['binary_data_dir']
        os.makedirs(self.binary_data_dir, exist_ok=True)
        self.build_phone_encoder()
        self.build_lang_map()
        self.build_spk_map()
        self.lr = LengthRegulator()
        self.pe = get_pitch_extractor_cls(hparams)(hparams)
        self.vocoder = get_vocoder_cls(hparams)(hparams)
        binarization_args = hparams["binarization_args"]
        if binarization_args['shuffle']:
            random.seed(3407)
            random.shuffle(self.transcription_item_list)

    def get_ph_name(self, ph, language):
        ph = f"{ph}/{language}"
        return self.ph2merged.get(ph, ph)

    def build_phone_encoder(self):
        hparams = self.hparams
        ph2merged = {}
        if hparams["merged_phoneme_dict"] is not None and hparams["merged_phoneme_dict"] != "":
            fn = f"{self.binary_data_dir}/{hparams['merged_phoneme_dict']}"
            f = open(fn, 'r')
            merge_dict = json.load(f)
            for merged, phs in merge_dict.items():
                for ph in phs:
                    ph2merged[ph] = merged
            f.close()
        self.ph2merged = ph2merged

        ph_set_fn = f"{self.binary_data_dir}/phone_set.json"
        ph_set = {}
        if not os.path.exists(ph_set_fn):
            for lang, dictionary in hparams["dictionary"].items():
                f = open(dictionary, 'r')
                for x in f.readlines():
                    ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                    for ph in ph_list:
                        ph_set[f"{ph}/{lang}"] = self.get_ph_name(ph, lang)
                f.close()
            json.dump(ph_set, open(ph_set_fn, 'w'))
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
        ph_list = list(sorted(ph_set.values()))
        print("| phone set: ", ph_list)
        self.ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")
    
    def build_spk_map(self):
        self.spk_ids = list(range(len(self.datasets)))
        self.spk_map = {ds["speaker"]: i for i, ds in enumerate(self.datasets)}
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{self.binary_data_dir}/spk_map.json"
        with open(spk_map_fn, 'w') as f:
            json.dump(self.spk_map, f)
    
    def build_lang_map(self):
        hparams = self.hparams
        self.lang_ids = list(range(len(self.datasets)))
        self.lang_map = {ds: i for i, ds in enumerate(hparams["dictionary"].keys())}
        print("| lang_map: ", self.lang_map)
        lang_map_fn = f"{hparams['binary_data_dir']}/lang_map.json"
        with open(lang_map_fn, 'w') as f:
            json.dump(self.lang_map, f)

    def load_meta_data(self):
        transcription_item_list = []
        for dataset in self.datasets:
            raw_data_dir, processed_data_dir = dataset["raw_data_dir"], dataset["processed_data_dir"]
            transcription_file = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in transcription_file.readlines():
                r = _r.split('|') # item_name | text | ph | dur_list | ph_num
                item_name = r[0]
                ph_text = [self.get_ph_name(p, dataset["language"]) for p in r[2].split(' ')]
                ph_seq = self.phone_encoder.encode(ph_text)
                lang_id = self.lang_map[dataset["language"]]
                item = {
                    "ph_seq" : ph_seq,
                    "ph_dur" : [float(x) for x in r[3].split(' ')],
                    "wav_fn" : f"{raw_data_dir}/wav/{item_name}.wav",
                    "spk_id" : self.spk_map[dataset["speaker"]],
                    "lang_id" : lang_id,
                    "lang_seq" : [lang_id]*len(ph_seq)
                }
                transcription_item_list.append(item)
            transcription_file.close()
        return transcription_item_list

    def process_item(self, item):
        item: dict
        hparams = self.hparams
        lr, pe = self.lr, self.pe

        wav, mel = self.vocoder.wav2spec(item.wav_fn, hparams=hparams)
        preprocessed_item = {
            "mel" : mel,
            "spk_id" : item.spk_id,
            "ph_seq" : np.array(item.ph_seq, dtype=np.int64),
            "ph_dur" : np.array(item.ph_dur, dtype=np.float32),
            "lang_seq" : np.array(item.lang_seq, dtype=np.int64),
            "sec" : len(wav) / hparams['audio_sample_rate'],
        }

        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        preprocessed_item["mel2ph"] = get_mel2ph_dur(lr, torch.FloatTensor(item.ph_dur), mel.shape[0], timestep)

        f0, uv = pe.get_pitch(
            wav, 
            samplerate = hparams['audio_sample_rate'], 
            length = mel.shape[0], 
            hop_size = hparams['hop_size'], 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item.item_name}, wav_fn: {item.wav_fn}"
        preprocessed_item["f0"] = f0

        return preprocessed_item