import os
import torch


class Binarizer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.datasets: dict = hparams['datasets']
        self.data_dir = os.path.join(hparams['data_dir'], self.category())
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_meta_data(self) -> list:
        raise NotImplementedError
    
    def process_item(self, item):
        raise NotImplementedError
    
    @staticmethod   
    def category():
        raise NotImplementedError
    
BINARIZERS = {}

def register_binarizer(cls):
    BINARIZERS[cls.category()] = cls
    return cls

def get_binarizer_cls(task):
    if task not in BINARIZERS:
        raise ValueError(f"Binarizer {task} not found in BINARIZERS")
    return BINARIZERS[task]