import os
import sys

sys.path.append(os.getcwd())
from utils.hparams import hparams, set_hparams

if __name__ == "__main__":
    set_hparams(config="preprocess/config.yaml", exp_name="test")
    datasets = hparams['dataset']
    print(datasets)