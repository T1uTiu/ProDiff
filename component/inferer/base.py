from typing import Dict

import torch


class Inferer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_model(self):
        raise NotImplementedError
    
    def run_model(self, **inp):
        raise NotImplementedError

INFERERS: Dict[str, type] = {}
def register_inferer(cls):
    INFERERS[cls.__name__.lower()] = cls
    INFERERS[cls.__name__] = cls
    return cls

def get_inferer_cls(hparams):
    cls_name = hparams['inferer'].lower()
    if cls_name not in INFERERS:
        raise ValueError(f"Inferer {cls_name} not found in INFERERS")
    return INFERERS[hparams['inferer']]