import numpy as np
from tqdm import tqdm

import utils
from component.binarizer.base import Binarizer
from utils.indexed_datasets import IndexedDatasetBuilder

class BinarizeHandler:
    def __init__(self, hparams):
        self.hparams = hparams
        binarizer_cls = hparams["binarizer_cls"]
        self.binarizer: Binarizer = utils.get_cls(binarizer_cls)(hparams)
        self.binary_data_dir = hparams['binary_data_dir']
        self.transcription_item_list = self.binarizer.load_meta_data()

    def get_transcription_item_list(self, prefix):
        hparams = self.hparams
        if prefix == 'valid':
            for i in range(0, hparams["test_num"]+hparams["valid_num"]):
                yield self.transcription_item_list[i]
        elif prefix == 'test':
            for i in range(0, hparams["test_num"]):
                yield self.transcription_item_list[i]
        else:
            for i in range(hparams["test_num"]+hparams["valid_num"], len(self.transcription_item_list)):
                yield self.transcription_item_list[i]

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(path=f'{data_dir}/{prefix}')
        lengths, f0s, total_sec = [], [], 0

        for item in tqdm(self.get_transcription_item_list(prefix)):
            preprocess_item = self.binarizer.process_item(item)
            builder.add_item(preprocess_item)
            if "sec" in preprocess_item:
                total_sec += preprocess_item["sec"]
            if "mel" in preprocess_item:
                lengths.append(preprocess_item["mel"].shape[0])
            if "f0" in preprocess_item:
                f0s.append(preprocess_item.f0)
        builder.finalize()
        
        if lengths > 0:
            np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        if total_sec > 0:
            print(f"| {prefix} total duration: {total_sec:.3f}s")
    
    def handle(self):
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')
