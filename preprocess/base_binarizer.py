import os
from typing import List

import torch

os.environ["OMP_NUM_THREADS"] = "1"

import json
import random
import traceback
from dataclasses import asdict, dataclass

import numpy as np
from tqdm import tqdm

from modules.fastspeech.tts_modules import LengthRegulator
from utils.data_gen_utils import (build_phone_encoder, get_mel2ph_dur,
                                       get_pitch)
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run
from vocoders.base_vocoder import VOCODERS

@dataclass
class TranscriptionItem:
    item_name: str
    text: str
    phoneme_raw: str
    duration: list
    wav_fn: str
    speaker_id: int
    
    phoneme: np.ndarray = None
    mel: np.ndarray = None
    mel_len: int = None
    mel2ph: np.ndarray = None
    sec: float = None
    f0: np.ndarray = None
    pitch: np.ndarray = None


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self):
        self.datasets: dict = hparams['datasets']
        self.raw_data_dirs = hparams['raw_data_dir']
        self.processed_data_dirs = hparams['processed_data_dir']
        self.binary_data_dir = hparams['binary_data_dir']
        self.binarization_args = hparams['binarization_args']
        os.makedirs(self.binary_data_dir, exist_ok=True)

        self.speakers = hparams['speakers']
        self.spk_map, self.spk_ids = self.build_spk_map()

        self.phone_encoder = self.build_phone_encoder()

        self.lr = LengthRegulator()
        self.load_meta_data()
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_items(self):
        return self.transcription_item_list[hparams['test_num']+hparams['valid_num']:]

    @property
    def valid_items(self):
        return self.transcription_item_list[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_items(self):
        return self.transcription_item_list[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'
    
    def load_meta_data(self):
        self.transcription_item_list: List[TranscriptionItem] = []
        for ds_id, dataset in enumerate(self.datasets):
            raw_data_dir, processed_data_dir = dataset["raw_data_dir"], dataset["processed_data_dir"]
            transcription_file = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in transcription_file.readlines():
                r = _r.split('|') # item_name | txt | ph | unknown | spk_id | dur_list
                item_name = raw_item_name =  r[0]
                if len(self.datasets) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                item = TranscriptionItem(
                    item_name=item_name,
                    text=r[1],
                    phoneme_raw=r[2],
                    duration=[float(x) for x in r[5].split(' ')],
                    wav_fn=f"{raw_data_dir}/wav/{raw_item_name}.wav",
                    speaker_id=self.spk_map[dataset["speaker"]]
                )
                self.transcription_item_list.append(item) 
            transcription_file.close()
    
    def get_transcription_item_list(self, prefix):
        if prefix == 'valid':
            items = self.valid_items
        elif prefix == 'test':
            items = self.test_items
        else:
            items = self.train_items
        return items

    def build_spk_map(self):
        spk_ids = list(range(len(self.speakers)))
        spk_map = {ds["speaker"]: i for i, ds in enumerate(self.datasets)}
        print("| spk_map: ", spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        with open(spk_map_fn, 'w') as f:
            json.dump(spk_map, f)
        return spk_map, spk_ids

    def build_phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = set(['AP', "SP"])
        if not os.path.exists(ph_set_fn):
            merge_ph = {}
            if "merge" in hparams["dictionary"]:
                f = open(hparams["dictionary"]["merge"], 'r')
                merge_dict = json.load(f)
                for merged, original in merge_dict.items():
                    merge_ph[original] = merged
                f.close()
            for dictionary in hparams[["dictionary"]]:
                with open(dictionary, 'r') as f:
                    for x in f.readlines():
                        ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                        for ph in ph_list:
                            if ph in merge_ph:
                                ph_set.add(merge_ph[ph])
                            else:
                                ph_set.add(ph)
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])


    def process(self):
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths, f0s, total_sec = [], [], 0 # 统计信息

        meta_data = self.get_transcription_item_list(prefix)
        args = [[m, self.phone_encoder, self.lr, hparams] for m in meta_data]

        num_workers = 2
        for _, (_, precessed_item) in enumerate(
                zip(tqdm(args), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if precessed_item is None:
                continue
            builder.add_item(asdict(precessed_item))

            total_sec += precessed_item.sec
            lengths.append(precessed_item.mel_len)
            f0s.append(precessed_item.f0)

        builder.finalize()

        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item: TranscriptionItem, encoder, lr, hparams):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(item.wav_fn, hparams=hparams)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(item.wav_fn)
        item.mel = mel
        item.sec = len(wav) / hparams['audio_sample_rate']
        item.mel_len = mel.shape[0]
        
        try:
            f0, pitch = get_pitch(wav, mel, hparams)
            if sum(f0) == 0:
                raise BinarizationError("Empty f0")
            item.f0 = f0
            item.pitch = pitch

            item.phoneme = encoder.encode(item.phoneme_raw)

            timestep = hparams['hop_size'] / hparams['audio_sample_rate']
            mel2ph = get_mel2ph_dur(lr, torch.FloatTensor(item.duration), item.mel_len, timestep)
            item.mel2ph = mel2ph
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item.item_name}, wav_fn: {item.wav_fn}")
            return None
        return item

if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
