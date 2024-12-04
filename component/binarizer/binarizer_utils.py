import json
import os

from utils.text_encoder import TokenTextEncoder


def build_phone_encoder(data_dir: str, hparams: dict):
    ph2merged = {}
    if hparams.get("merged_phoneme_dict", None) is not None :
        fn = f"{data_dir}/{hparams['merged_phoneme_dict']}"
        f = open(fn, 'r')
        merge_dict = json.load(f)
        for merged, phs in merge_dict.items():
            for ph in phs:
                ph2merged[ph] = merged
        f.close()

    ph_set_fn = f"{data_dir}/phone_set.json"
    ph_map = {}
    if not os.path.exists(ph_set_fn):
        for lang, dictionary in hparams["dictionary"].items():
            f = open(dictionary, 'r')
            for x in f.readlines():
                ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                for ph in ph_list:
                    ph = f"{ph}/{lang}"
                    ph_map[ph] = ph2merged.get(ph, ph)
            f.close()
        json.dump(ph_map, open(ph_set_fn, 'w'))
    else:
        ph_map = json.load(open(ph_set_fn, 'r'))
    ph_list = list(sorted(ph_map.values()))
    print("| phone set: ", ph_list)
    ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")
    return ph_map, ph_encoder