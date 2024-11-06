import sys, os
sys.path.append(os.getcwd())

os.environ["OMP_NUM_THREADS"] = "1"

import importlib
from utils.hparams import set_hparams, hparams


def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer').split(".")
    pkg, cls_name = ".".join(binarizer_cls[:-1]), binarizer_cls[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    set_hparams()
    binarize()
