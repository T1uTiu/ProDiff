import os
import json
import textgrid
import pickle
import librosa
from tqdm import tqdm

class PreprocessHandler:
    def __init__(self, data_dir, lang):
        self.data_dir = data_dir    
        self.lang = lang

    def textgrid_to_label(self):
        tg_dir = f"{self.data_dir}/TextGrid"
        label = {}
        for tg_fn in tqdm(os.listdir(tg_dir)):
            item = {}
            tg = textgrid.TextGrid.fromFile(f"{tg_dir}/{tg_fn}")
            name = tg_fn.replace(".TextGrid", "")
            ph_tier = tg.getFirst("phone")
            ph_seq, ph_dur = [], []
            for x in ph_tier:
                ph_seq.append(x.mark)
                ph_dur.append(f"{x.maxTime - x.minTime:.4f}")
            item["ph_seq"] = " ".join(ph_seq)
            item["ph_dur"] = " ".join(ph_dur)
            label[name] = item
        return label

    def add_ph_num_label(self, labels, override=False):
        dictionary_fn = f"dictionary/{self.lang}_phones.txt"
        c_set, v_set = set(), set(["AP", "SP"])
        with open(dictionary_fn, 'r', encoding='utf-8') as f:
            for x in  f.readlines():
                line = x.split("\n")[0].split(' ')
                ph, ph_type = line[0], line[1]
                if ph_type == "consonant":
                    c_set.add(ph)
                else:
                    v_set.add(ph)
        for label in tqdm(labels.values()):
            if "ph_num" in label and not override:
                continue
            ph_num = []
            for i, ph in enumerate(label["ph_seq"].split(" ")):
                if ph in v_set or i == 0:
                    ph_num.append(1)
                else:
                    ph_num[-1] += 1
            label["ph_num"] = " ".join(map(str, ph_num))

    def cal_note_seq(self, note_midi: float, note_rest: bool):
        midi_num = round(note_midi, 0)
        cent = int(round(note_midi - midi_num, 2) * 100)
        if cent > 0:
            cent = f"+{cent}"
        elif cent == 0:
            cent = ""
        seq = f"{librosa.midi_to_note(midi_num, unicode=False)}{cent}"
        return seq if not note_rest else "rest"

    def add_note_midi_label(self, labels, override=False):
        rawmidi_dir = f"{self.data_dir}/midi"
        for item_name, label in tqdm(labels.items()):
            if "note_seq" in label and not override:
                continue
            with open(f"{rawmidi_dir}/{item_name}.rawmid", "rb") as f:
                raw_midi = pickle.loads(f.read())
            note_midi = raw_midi["note_midi"]
            note_rest = raw_midi["note_rest"]
            note_seq = [self.cal_note_seq(midi, rest) for midi, rest in zip(note_midi, note_rest)]
            note_dur = [f"{x:.4f}" for x in raw_midi["note_dur"]]
            label["note_seq"] = " ".join(note_seq)
            label["note_dur"] = " ".join(note_dur)
            

    def handle(self, extract_note=False, override_ph_num=False, override_note_midi=False, override_ori_label=False):
        tgt_label_fn = f"{self.data_dir}/label.json" if override_ori_label else f"{self.data_dir}/label_new.json"
        print("1. build label.json")
        if os.path.exists(f"{self.data_dir}/label.json"):
            print("label.json already exists, skip textgrid_to_label")
            with open(f"{self.data_dir}/label.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
        else:
            print("build label.json from TextGrid")
            labels = self.textgrid_to_label()
        if not extract_note:
            with open(tgt_label_fn, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=4, ensure_ascii=False)
            print("preprocess is done, label.json is saved")
            return
        print("2. add ph_num to label.json")
        if all("ph_num" in label for label in labels.values()) and not override_ph_num:
            print("ph_num already exists, skip add_ph_num_label")
        else:
            if self.lang not in ["zh", "jp"]:
                print("auto process only support zh and jp, exit")
                return
            self.add_ph_num_label(labels, override_ph_num)
        print("3. add note_midi to label.json")
        if all("note_seq" in label for label in labels.values()) and not override_note_midi:
            print("note_seq already exists, skip add_note_midi_label")
        else:
            self.add_note_midi_label(labels, override_note_midi)
        print("preprocess is done, saving label.json")
        with open(tgt_label_fn, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=4, ensure_ascii=False)
        