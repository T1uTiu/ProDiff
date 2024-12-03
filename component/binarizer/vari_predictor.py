import numpy as np
import torch
from component.binarizer.base import Binarizer
from component.binarizer.binarizer_utils import build_phone_encoder
from modules.fastspeech.tts_modules import LengthRegulator


class DurPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.ph2merged, self.phone_encoder = build_phone_encoder(hparams)
        self.lr = LengthRegulator()

    
    def load_meta_data(self) -> list:
        transcription_item_list = []
        for dataset in self.datasets:
            processed_data_dir = dataset["processed_data_dir"]
            transcription_file = open(f"{processed_data_dir}/transcriptions.txt", 'r', encoding='utf-8')
            for _r in transcription_file.readlines():
                r = _r.split('|') # item_name | text | ph | dur_list | ph_num
                ph_text = [self.get_ph_name(p, dataset["language"]) for p in r[2].split(' ')]
                ph_seq = self.phone_encoder.encode(ph_text)
                item = {
                    "ph_seq" : ph_seq,
                    "ph_dur" : [float(x) for x in r[3].split(' ')],
                    "ph_num" : [int(x) for x in r[4].split(' ')]
                }
                transcription_item_list.append(item)
            transcription_file.close()
        return transcription_item_list
    
    def process_item(self, item):
        ph_num = torch.LongTensor(item["ph_num"])
        word_num = ph_num.shape[0]
        ph2word = self.lr(ph_num[None])[0]
        # onset指哪些音素是音符的开头
        onset = torch.diff(ph2word, dim=0, prepend=ph2word.new_zeros(1)) > 0
        # 音素持续时间
        ph_dur = np.array(item["ph_dur"], dtype=np.float32)
        # 音符持续时间
        word_dur = ph_dur.new_zeros(word_num+1).scatter_add(
            0, ph2word, ph_dur
        )[1:]
        preprocessed_item = {
            "ph_seq": np.array(item["ph_seq"], dtype=np.int64),
            "ph_dur": ph_dur,
            "word_dur": word_dur,
            "onset": onset,
        }
        return preprocessed_item