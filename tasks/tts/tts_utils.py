import importlib

from preprocess.base_binarizer import BaseBinarizer
from preprocess.txt_processors.base_text_processor import get_txt_processor_cls
from utils.hparams import hparams


def parse_dataset_configs():
    max_tokens = hparams['max_tokens']
    max_sentences = hparams['max_sentences']
    max_valid_tokens = hparams['max_valid_tokens']
    if max_valid_tokens == -1:
        hparams['max_valid_tokens'] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams['max_valid_sentences']
    if max_valid_sentences == -1:
        hparams['max_valid_sentences'] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences


def parse_mel_losses():
    mel_losses = hparams['mel_losses'].split("|")
    loss_and_lambda = {}
    for i, l in enumerate(mel_losses):
        if l == '':
            continue
        if ':' in l:
            l, lbd = l.split(":")
            lbd = float(lbd)
        else:
            lbd = 1.0
        loss_and_lambda[l] = lbd
    print("| Mel losses:", loss_and_lambda)
    return loss_and_lambda

def load_data_binarizer():
    binarizer_cls = hparams['binarizer_cls']
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer: BaseBinarizer = getattr(importlib.import_module(pkg), cls_name)()
    binarization_args = {}
    binarization_args.update(hparams['binarization_args'])
    return binarizer, binarization_args