import os
import numpy as np
from tqdm import tqdm

from component.binarizer import get_binarizer_cls
from component.binarizer.base import Binarizer
from utils.indexed_datasets import IndexedDatasetBuilder

class BinarizeHandler:
    def __init__(self, hparams):
        self.hparams = hparams
        self.binarizer: Binarizer = get_binarizer_cls(hparams["task"])(hparams)
        self.binary_data_dir = os.path.join(hparams['data_dir'], self.binarizer.category())
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
            preprocessed_item = self.binarizer.process_item(item)
            builder.add_item(preprocessed_item)
            if "sec" in preprocessed_item:
                total_sec += preprocessed_item["sec"]
            assert "length" in preprocessed_item, "Preprocessed item must have 'length' field"
            lengths.append(preprocessed_item["length"])
            if "f0" in preprocessed_item:
                f0s.append(preprocessed_item["f0"])
        builder.finalize()
        
        if len(lengths) > 0:
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
