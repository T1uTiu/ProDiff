import os

import torch
os.environ["OMP_NUM_THREADS"] = "1"

from modules.fastspeech.tts_modules import LengthRegulator
from utils.multiprocess_utils import chunked_multiprocess_run
import random
import traceback
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen.tts.data_gen_utils import get_mel2ph, get_mel2ph_dur, get_pitch, build_phone_encoder
from utils.hparams import set_hparams, hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import VOCODERS
import pandas as pd


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['preprocess_args']
        self.lr = LengthRegulator()
        self.load_meta_data()
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    def load_meta_data(self):
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2spk = {}
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = pd.read_csv(f"{processed_data_dir}/metadata_phone.csv", dtype=str)
            for r_idx, r in self.meta_df.iterrows():
                item_name = r['item_name']
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = r['txt']
                self.item2ph[item_name] = r['ph']
                self.item2wavfn[item_name] = os.path.join(hparams['raw_data_dir'], 'wavs', os.path.basename(r['wav_fn']).split('_')[1])
                self.item2spk[item_name] = r.get('spk', 'SPK1')
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
        self.item_names = sorted(list(self.item2txt.keys()))

    @property
    def train_item_names(self):
        return self.item_names[hparams['test_num']+hparams['valid_num']:]

    @property
    def valid_item_names(self):
        return self.item_names[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_item_names(self):
        return self.item_names[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'

    def build_spk_map(self):
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.item2spk[item_name]
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.item2spk[item_name]]

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for processed_data_dir in self.processed_data_dirs:
                ph_set += [x.split(' ')[0] for x in open(f'{processed_data_dir}/dict.txt').readlines()]
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
        print("| phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

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
            tg_fn = self.item2tgfn.get(item_name) # TextGrid file name
            wav_fn = self.item2wavfn[item_name] # Audio file name
            spk_id = self.item_name2spk_id(item_name)
            yield item_name, ph, txt, tg_fn, wav_fn, spk_id

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [self.phone_encoder, self.lr, hparams])
        # num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3)) # 线程个数
        num_workers = 2
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if self.binarization_args['with_spk_embed'] else None
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
    def process_item(cls, item_name, ph, dur, txt, wav_fn, spk_id, encoder, lr, hparams):
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
            'len': mel.shape[0], 
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
