from component.binarizer.base import Binarizer


class DurPredictorBinarizer(Binarizer):
    def __init__(self, hparams):
        super().__init__(hparams)

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
        return super().process_item(item)