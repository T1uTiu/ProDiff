import os

import torch
os.environ["OMP_NUM_THREADS"] = "1"

from modules.fastspeech.tts_modules import LengthRegulator
from utils.multiprocess_utils import chunked_multiprocess_run
import random
import traceback
import json
from tqdm import tqdm
from data_gen.tts.data_gen_utils import get_mel2ph_dur, get_pitch, build_phone_encoder
from utils.hparams import set_hparams, hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import VOCODERS


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self):
        self.processed_data_dirs = hparams['processed_data_dir']
        self.raw_data_dirs = hparams['raw_data_dir']
        self.binary_data_dir = hparams['binary_data_dir']
        os.makedirs(self.binary_data_dir, exist_ok=True)
        self.binarization_args = hparams['binarization_args']

        self.speakers = hparams['speakers']
        self.spk_map, self.spk_ids = self.build_spk_map()

        self.phone_encoder = self.build_phone_encoder()

        self.lr = LengthRegulator()
        self.load_meta_data()
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        return self.item_names[hparams['test_num']+hparams['valid_num']:]

    @property
    def valid_item_names(self):
        return self.item_names[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_item_names(self):
        return self.item_names[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'
    
    def load_meta_data(self):
        self.item2txt = {}
        self.item2ph = {}
        self.item2dur = {}
        self.item2wavfn = {}
        self.item2spk = {}
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in self.meta_df.readlines():
                r = _r.split('|') # item_name | txt | ph | unknown | spk_id | dur_list
                item_name = raw_item_name =  r[0]
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = r[1]
                self.item2ph[item_name] = r[2]
                self.item2wavfn[item_name] = f"{self.raw_data_dirs[ds_id]}/wav/{raw_item_name}.wav"
                self.item2spk[item_name] = self.speakers[ds_id]
                self.item2dur[item_name] = [float(x) for x in r[5].split(' ')]
        self.item_names = sorted(list(self.item2txt.keys()))
    
    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        key_shift = int(self.binarization_args.get('key_shift', -1))
        key_shifts = [-key_shift, 0, key_shift] if key_shift != -1 else [0]
        for ks in key_shifts:
            for item_name in item_names:
                ph = self.item2ph[item_name] # Phoneme
                txt = self.item2txt[item_name] # Text
                wav_fn = self.item2wavfn[item_name] # Audio file name
                spk_id = self.item_name2spk_id(item_name)
                dur = self.item2dur[item_name]

                yield item_name, ph, dur, txt, wav_fn, spk_id, ks

    def build_spk_map(self):
        spk_ids = list(range(len(self.speakers)))
        spk_map = {x: i for i, x in zip(spk_ids, self.speakers)}
        print("| spk_map: ", spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(spk_map, open(spk_map_fn, 'w'))
        return spk_map, spk_ids

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.item2spk[item_name]]

    def build_phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = ['AP', "SP"]
        if not os.path.exists(ph_set_fn):
            for x in open(hparams['dictionary']).readlines():
                ph_set += x.split("\n")[0].split('\t')[1].split(' ') 
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

        meta_data = self.meta_data(prefix)
        args = [list(m) + [self.phone_encoder, self.lr, hparams] for m in meta_data]

        num_workers = 2
        for f_id, (_, item) in enumerate(
                zip(tqdm(args), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            item['spk_embed'] =  None
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
            f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item_name, ph, dur, txt, wav_fn, spk_id, key_shift, encoder, lr, hparams):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn, hparams=hparams)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 
            'txt': txt, 
            'ph': ph, 
            'dur': dur, 
            'mel': mel,  
            'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], # 真实时长
            'len': mel.shape[0], # 梅尔帧数
            'spk_id': spk_id
        }
        try:
            # get ground truth f0
            cls.get_pitch(wav, mel, res, hparams)
            try:
                res['phone'] = encoder.encode(ph)
            except:
                traceback.print_exc()
                raise BinarizationError(f"Empty phoneme")
            # get ground truth dur
            cls.get_align(dur, mel, lr, res, hparams)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

    @staticmethod
    def get_align(dur, mel, lr, res, hparams):
        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        res['mel2ph'] = get_mel2ph_dur(lr, torch.FloatTensor(dur), mel.shape[0], timestep)
        res['dur'] = dur

    @staticmethod
    def get_pitch(wav, mel, res, hparams):
        f0, pitch_coarse = get_pitch(wav, mel, hparams)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        res['f0'] = f0
        res['pitch'] = pitch_coarse

    @staticmethod
    def get_f0cwt(f0, res):
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        if np.any(np.isnan(Wavelet_lf0)):
            raise BinarizationError("NaN CWT")
        res['cwt_spec'] = Wavelet_lf0
        res['cwt_scales'] = scales
        res['f0_mean'] = logf0s_mean_org
        res['f0_std'] = logf0s_std_org


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
