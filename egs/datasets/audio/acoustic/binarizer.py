import json

from data_gen.tts.base_binarizer import BaseBinarizer
from data_gen.tts.data_gen_utils import build_phone_encoder
from modules.fastspeech.tts_modules import LengthRegulator
from utils.hparams import hparams
import os
import random

class AcousticBinarizer(BaseBinarizer):
    def load_meta_data(self):
        self.item2txt = {}
        self.item2ph = {}
        self.item2dur = {}
        self.item2wavfn = {}
        self.item2tgfn = {} # TextGrid file name
        self.item2spk = {}
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in self.meta_df.readlines():
                r = _r.split('|')
                item_name = raw_item_name = r[0]
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = r[1]
                self.item2ph[item_name] = r[2]
                self.item2wavfn[item_name] = f"{hparams['raw_data_dir']}/wav/{raw_item_name}.wav"
                self.item2spk[item_name] = 'SPK1'
                self.item2dur[item_name] = [float(x) for x in r[5].split(' ')]
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
        self.item_names = sorted(list(self.item2txt.keys()))

    # override
    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            ph = self.item2ph[item_name] # Phoneme
            txt = self.item2txt[item_name] # Text
            wav_fn = self.item2wavfn[item_name] # Audio file name
            spk_id = self.item_name2spk_id(item_name)
            dur = self.item2dur[item_name]
            yield item_name, ph, dur, txt, wav_fn, spk_id

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = ['AP', "SP"]
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for processed_data_dir in self.processed_data_dirs:
                for x in open(f'{processed_data_dir}/dict.txt').readlines():
                    ph_set += x.split("\n")[0].split('\t')[1].split(' ') 
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])
    
    