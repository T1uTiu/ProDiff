import os
from typing import List
import torch

os.environ["OMP_NUM_THREADS"] = "1"

import json
import random
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from modules.fastspeech.tts_modules import LengthRegulator
from component.pitch_extractor import get_pitch_extractor_cls
from utils.data_gen_utils import build_phone_encoder, get_mel2ph_dur
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import get_vocoder_cls

@dataclass
class TranscriptionItem:
    wav_fn: str
    spk_id: int
    lang_id: int
    ph_seq: List[int]
    ph_dur: List[float]
    lang_seq: List[int]

@dataclass
class PreprocessedItem:
    mel: np.ndarray = None

    spk_id: int = -1
    ph_seq: np.ndarray = None
    ph_dur: np.ndarray = None
    lang_seq: np.ndarray = None
    mel2ph: np.ndarray = None
    f0: np.ndarray = None

    sec: float = -1

    @property
    def mel_len(self):
        return self.mel.shape[0]


class BinarizeHandler:
    def __init__(self, hparams):
        self.hparams = hparams
        self.datasets: dict = hparams['datasets']
        self.binary_data_dir = hparams['binary_data_dir']
        self.binarization_args = hparams['binarization_args']
        os.makedirs(self.binary_data_dir, exist_ok=True)

        self.spk_map, self.spk_ids = self.build_spk_map()

        self.phone_encoder = self.build_phone_encoder()

        self.lang_map, self.lang_ids = self.build_lang_map()

        self.lr = LengthRegulator()
        self.pitch_extractor = get_pitch_extractor_cls(hparams)(hparams)
        self.vocoder = get_vocoder_cls(hparams)(hparams)
        self.load_meta_data()
        if self.binarization_args['shuffle']:
            random.seed(3407)
            random.shuffle(self.transcription_item_list)

    @property
    def train_items(self):
        return self.transcription_item_list[hparams['test_num']+hparams['valid_num']:]

    @property
    def valid_items(self):
        return self.transcription_item_list[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_items(self):
        return self.transcription_item_list[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'

    def get_ph_name(self, ph, language):
        ph = f"{ph}/{language}"
        return ph if ph not in self.ph2merged else self.ph2merged[ph]

    def load_meta_data(self):
        self.transcription_item_list: List[TranscriptionItem] = []
        for dataset in self.datasets:
            raw_data_dir, processed_data_dir = dataset["raw_data_dir"], dataset["processed_data_dir"]
            transcription_file = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in transcription_file.readlines():
                r = _r.split('|') # item_name | text | ph | dur_list
                item_name = r[0]
                ph_text = [self.get_ph_name(p, dataset["language"]) for p in r[2].split(' ')]
                ph_seq = self.phone_encoder.encode(ph_text)
                lang_id = self.lang_map[dataset["language"]]
                item = TranscriptionItem(
                    ph_seq = ph_seq,
                    ph_dur = [float(x) for x in r[-1].split(' ')],
                    wav_fn = f"{raw_data_dir}/wav/{item_name}.wav",
                    spk_id = self.spk_map[dataset["speaker"]],
                    lang_id = lang_id,
                    lang_seq = [lang_id]*len(ph_seq)
                )
                self.transcription_item_list.append(item) 
            transcription_file.close()
    
    def get_transcription_item_list(self, prefix):
        if prefix == 'valid':
            for i in range(0, hparams["test_num"]+hparams["valid_num"]):
                yield self.transcription_item_list[i]
        elif prefix == 'test':
            for i in range(0, hparams["test_num"]):
                yield self.transcription_item_list[i]
        else:
            for i in range(hparams["test_num"]+hparams["valid_num"], len(self.transcription_item_list)):
                yield self.transcription_item_list[i]

    def build_spk_map(self):
        spk_ids = list(range(len(self.datasets)))
        spk_map = {ds["speaker"]: i for i, ds in enumerate(self.datasets)}
        print("| spk_map: ", spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        with open(spk_map_fn, 'w') as f:
            json.dump(spk_map, f)
        return spk_map, spk_ids
    
    def build_lang_map(self):
        lang_ids = list(range(len(self.datasets)))
        lang_map = {ds: i for i, ds in enumerate(hparams["dictionary"].keys())}
        print("| lang_map: ", lang_map)
        lang_map_fn = f"{hparams['binary_data_dir']}/lang_map.json"
        with open(lang_map_fn, 'w') as f:
            json.dump(lang_map, f)
        return lang_map, lang_ids

    def build_phone_encoder(self):
        ph2merged = {}
        if hparams["merged_phoneme_dict"] is not None and hparams["merged_phoneme_dict"] != "":
            fn = f"{hparams['binary_data_dir']}/{hparams['merged_phoneme_dict']}"
            f = open(fn, 'r')
            merge_dict = json.load(f)
            for merged, phs in merge_dict.items():
                for ph in phs:
                    ph2merged[ph] = merged
            f.close()
        self.ph2merged = ph2merged

        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = {
            "c": set(['AP', "SP"]),
            "v": set()
        }
        if not os.path.exists(ph_set_fn):
            for lang, dictionary in hparams["dictionary"].items():
                f = open(dictionary, 'r')
                for x in f.readlines():
                    ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                    for i, ph in enumerate(ph_list):
                        ph = self.get_ph_name(ph, lang)
                        if len(ph_list) == 1 or i == 1:
                            ph_set["v"].add(ph)
                        else:
                            ph_set["c"].add(ph)
                f.close()
            ph_set = list(sorted(ph_set["c"])) + list(sorted(ph_set["v"]))
            json.dump(ph_set, open(ph_set_fn, 'w'))
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(path=f'{data_dir}/{prefix}')
        lengths, f0s, total_sec = [], [], 0 # 统计信息


        for item in tqdm(self.get_transcription_item_list(prefix)):
            preprocess_item = self.process_item(item)
            builder.add_item(preprocess_item)

            total_sec += preprocess_item.sec
            lengths.append(preprocess_item.mel_len)
            f0s.append(preprocess_item.f0)


        builder.finalize()

        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item: TranscriptionItem):
        lr, pe = self.lr, self.pitch_extractor
        wav, mel = self.vocoder.wav2spec(item.wav_fn, hparams=hparams)

        preprocessed_item = PreprocessedItem(
            mel = mel,
            spk_id = item.spk_id,
            ph_seq = np.array(item.ph_seq, dtype=np.int64),
            ph_dur = np.array(item.ph_dur, dtype=np.float32),
            lang_seq = np.array(item.lang_seq, dtype=np.int64),
            sec = len(wav) / hparams['audio_sample_rate'],
        )

        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        preprocessed_item.mel2ph = get_mel2ph_dur(lr, torch.FloatTensor(item.ph_dur), preprocessed_item.mel_len, timestep)

        f0, uv = pe.get_pitch(
            wav, 
            samplerate = hparams['audio_sample_rate'], 
            length = mel.shape[0], 
            hop_size = hparams['hop_size'], 
            interp_uv = hparams['interp_uv']
        )
        assert not uv.all(), f"all unvoiced. item_name: {item.item_name}, wav_fn: {item.wav_fn}"
        preprocessed_item.f0 = f0

        return preprocessed_item
    
    def handle(self):
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')
