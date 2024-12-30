import json
import os
import numpy as np
import torch
from torch.functional import F
from component.binarizer.base import Binarizer, register_binarizer
from component.binarizer.binarizer_utils import build_phone_encoder
from modules.fastspeech.tts_modules import LengthRegulator

@register_binarizer
class DurPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.ph_map, self.ph_encoder = build_phone_encoder(self.data_dir, hparams["dictionary"])
        self.lr = LengthRegulator()

    @staticmethod
    def category():
        return "dur"

    def load_meta_data(self) -> list:
        transcription_item_list = []
        for dataset in self.datasets:
            data_dir = dataset["data_dir"]
            lang = dataset["language"]
            with open(f"{data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            for label in labels:
                ph_text = [self.ph_map[f"{p}/{lang}"] for p in label["ph_seq"].split(' ')]
                ph_seq = self.ph_encoder.encode(ph_text)
                item = {
                    "ph_seq" : ph_seq,
                    "ph_dur" : [float(x) for x in label["ph_dur"].split(' ')],
                    "ph_num" : [int(x) for x in label["ph_num"].split(' ')]
                }
                transcription_item_list.append(item)
        return transcription_item_list
    
    def process_item(self, item):
        ph_num = torch.LongTensor(item["ph_num"])
        word_num = ph_num.shape[0]
        ph2word = self.lr(ph_num[None])[0]
        # onset指哪些音素是音符的开头
        onset = torch.diff(ph2word, dim=0, prepend=ph2word.new_zeros(1))
        # 音素持续时间
        ph_dur = torch.FloatTensor(item["ph_dur"])
        # 音符持续时间
        word_dur = ph_dur.new_zeros(word_num+1).scatter_add(
            0, ph2word, ph_dur
        )[1:]
        word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 0, ph2word) # T_w => T_ph
        preprocessed_item = {
            "ph_seq": np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur": ph_dur.cpu().numpy(),
            "word_dur": word_dur.cpu().numpy(),
            "onset": onset.cpu().numpy(),
            "length": len(item["ph_seq"])
        }
        return preprocessed_item