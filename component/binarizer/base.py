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