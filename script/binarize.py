import os
import sys

sys.path.append(os.getcwd())

import click

os.environ["OMP_NUM_THREADS"] = "1"

from preprocess.base_binarizer import BaseBinarizer
from utils.hparams import hparams, set_hparams


@click.command()
@click.option("--config", type=str, required=True)
def binarize(config):
    set_hparams(config=config)
    binarizer = BaseBinarizer()
    binarizer.process()


if __name__ == '__main__':
    binarize()
