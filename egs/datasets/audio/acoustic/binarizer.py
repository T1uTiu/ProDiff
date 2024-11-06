from data_gen.tts.base_binarizer import BaseBinarizer

class AcousticBinarizer(BaseBinarizer):
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

    # override
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
    
    