import os, sys
sys.path.append(os.getcwd())

from preprocess.data_gen_utils import get_pitch



import torch

from utils.audio import save_wav
from vocoders.base_vocoder import VOCODERS
import click

from utils.hparams import hparams, set_hparams

@click.group()
def vocode():
    pass

@vocode.command()
@click.argument("wav", type=str)
@click.option("--config", type=str, required=True)
def wav2wav(wav, config):
    set_hparams(config=config, exp_name='vocoder')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder = VOCODERS[hparams['vocoder']]()
    vocoder.to_device(device)
    # wav2spec
    wave, mel = vocoder.wav2spec(wav, hparams=hparams)
    # spec2wav
    f0, _ = get_pitch(wave, mel, hparams)
    res = vocoder.spec2wav(mel, f0=f0)
    title = os.path.basename(wav).split('.')[0]
    save_wav(res, f'infer_out/{title}.wav', hparams['audio_sample_rate'])

@vocode.command()
@click.argument("spec", type=str)
@click.option("--config", type=str, required=True)
def spec2wav(spec, config):
    pass


if __name__ == '__main__':
    vocode()
