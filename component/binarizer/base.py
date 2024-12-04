import torch


class Binarizer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.datasets: dict = hparams['datasets']
    
    def load_meta_data(self) -> list:
        raise NotImplementedError
    
    def process_item(self, item):
        raise NotImplementedError
    
    @staticmethod   
    def category(cls):
        raise NotImplementedError
    
BINARIZERS = {}

def register_binarizer(cls):
    BINARIZERS[cls.category()] = cls
    return cls

def get_binarizer_cls(hparams):
    binarizer = hparams["task"]
    if binarizer not in BINARIZERS:
        raise ValueError(f"Binarizer {binarizer} not found in BINARIZERS")
    return BINARIZERS[binarizer]