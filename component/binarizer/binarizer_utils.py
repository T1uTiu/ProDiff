import csv
import json
import os

from utils.text_encoder import TokenTextEncoder


def build_phone_encoder(data_dir: str, hparams: dict):
    ph2global = {}
    if hparams["dictionary"].get("global", None):
        f = open(hparams["dictionary"]["global"], 'r')
        for label in csv.DictReader(f):
            for lang, ph in label.items():
                if lang == "global":
                    continue
                ph2global[f"{ph}/{lang}"] = label["global"]
        f.close()

    ph_set_fn = f"{data_dir}/phone_set.json"
    ph_map = {}
    if not os.path.exists(ph_set_fn):
        for lang, dictionary in hparams["dictionary"].items():
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