import csv
import json
import os

import numpy as np
import torch
from utils.text_encoder import TokenTextEncoder


def build_phone_encoder(data_dir: str, dictionary: dict):
    ph2global = {}
    if dictionary.get("global", None):
        f = open(dictionary["global"], 'r')
        for label in csv.DictReader(f):
            for lang, ph in label.items():
                if lang == "global":
                    continue
                ph2global[f"{ph}/{lang}"] = label["global"]
        f.close()

    ph_set_fn = f"{data_dir}/phone_set.json"
    ph_map = {}
    if not os.path.exists(ph_set_fn):
        for lang, dictionary in dictionary.items():
            if lang == "global":
                continue
            f = open(dictionary, 'r')
            ph_map[f"AP/{lang}"] = "AP"
            ph_map[f"SP/{lang}"] = "SP"
            for x in f.readlines():
                ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                for ph in ph_list:
                    ph = f"{ph}/{lang}"
                    ph_map[ph] = ph2global.get(ph, ph)
            f.close()
        json.dump(ph_map, open(ph_set_fn, 'w'))
    else:
        ph_map = json.load(open(ph_set_fn, 'r'))
    ph_list = list(sorted(set(ph_map.values())))
    print("| phone set: ", ph_list)
    ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")
    return ph_map, ph_encoder

def build_lang_map(data_dir, dictionary: dict):
    lang_map = {dt: i for i, dt in enumerate(dictionary.keys()) if dt != "global"}
    print("| lang_map: ", lang_map)
    lang_map_fn = f"{data_dir}/lang_map.json"
    with open(lang_map_fn, 'w') as f:
        json.dump(lang_map, f)
    return lang_map\
    
def build_spk_map(data_dir, datasets):
    spk_map = {ds["speaker"]: i for i, ds in enumerate(datasets)}
    print("| spk_map: ", spk_map)
    spk_map_fn = f"{data_dir}/spk_map.json"
    with open(spk_map_fn, 'w') as f:
        json.dump(spk_map, f)
    return spk_map