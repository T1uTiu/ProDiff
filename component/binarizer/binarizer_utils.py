import json
import os

from utils.text_encoder import TokenTextEncoder


def build_phone_encoder(self, hparams):
    binary_data_dir = hparams['binary_data_dir']
    ph2merged = {}
    if hparams["merged_phoneme_dict"] is not None and hparams["merged_phoneme_dict"] != "":
        fn = f"{binary_data_dir}/{hparams['merged_phoneme_dict']}"
        f = open(fn, 'r')
        merge_dict = json.load(f)
        for merged, phs in merge_dict.items():
            for ph in phs:
                ph2merged[ph] = merged
        f.close()

    ph_set_fn = f"{binary_data_dir}/phone_set.json"
    ph_set = {}
    if not os.path.exists(ph_set_fn):
        for lang, dictionary in hparams["dictionary"].items():
            f = open(dictionary, 'r')
            for x in f.readlines():
                ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                for ph in ph_list:
                    ph_set[f"{ph}/{lang}"] = self.get_ph_name(ph, lang)
            f.close()
        json.dump(ph_set, open(ph_set_fn, 'w'))
    else:
        ph_set = json.load(open(ph_set_fn, 'r'))
    ph_list = list(sorted(ph_set.values()))
    print("| phone set: ", ph_list)
    ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")
    return ph2merged, ph_encoder