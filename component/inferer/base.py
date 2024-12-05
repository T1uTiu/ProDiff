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
    
    @staticmethod
    def category():
        raise NotImplementedError

INFERERS: Dict[str, type] = {}
def register_inferer(cls):
    INFERERS[cls.category()] = cls
    return cls

def get_inferer_cls(task):
    if task not in INFERERS:
        raise ValueError(f"Inferer {task} not found in INFERERS")
    return INFERERS[task]