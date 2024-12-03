import os
from typing import Dict
import click
import torch

from handler.binarize import BinarizeHandler
from handler.infer import InferHandler
from component.train_task import ProDiffTask, ProDiffTeacherTask
from utils.data_gen_utils import get_pitch
from tasks.base_task import BaseTask
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams
from utils.pitch_utils import shift_pitch
from vocoders.base_vocoder import VOCODERS

@click.group()
def main():
    pass

@main.command()
@click.option("--config", type=str, required=True)
@click.option("--exp_name", type=str, required=True)
def binarize(config, exp_name):
    set_hparams(config=config, exp_name=exp_name)
    BinarizeHandler(hparams=hparams).handle()

trainer_map: Dict[str, BaseTask] = {
    "teacher": ProDiffTeacherTask,
}

@main.command()
@click.argument("trainer", type=str)
@click.option("--config", type=str, required=True)
@click.option("--exp_name", type=str, required=True)
def train(trainer, config, exp_name):
    assert trainer in trainer_map, f"Invalid trainer: {trainer}, use one of {list(trainer_map.keys())}"
    set_hparams(config=config, exp_name=exp_name)
    trainer_instance = trainer_map[trainer]
    trainer_instance.start()

inferer_map: Dict[str, str] = {
    "teacher": "ProDiffTeacherInferrer"
}

@main.command()
@click.argument("inferer", type=str)
@click.argument("proj", type=str)
@click.option("--config", type=str)
@click.option("--exp_name", type=str)
@click.option("--spk_name", type=str)
@click.option("--lang", type=str, default='zh')
@click.option("--keyshift", type=int, default=0)
def infer(inferer, proj, config, exp_name, spk_name, lang, keyshift):
    assert inferer in inferer_map, f"Invalid inferer: {inferer}, use one of {list(inferer_map.keys())}"
    set_hparams(config=config, exp_name=exp_name, spk_name=spk_name)
    hparams.setdefault("inferer", inferer_map[inferer])
    InferHandler(hparams=hparams).handle(None, proj, lang, keyshift)

@main.group()
def vocode():
    pass

@vocode.command()
@click.argument("wav", type=str)
@click.option("--config", type=str, required=True)
@click.option("--keyshift", type=int, default=0, required=False)
@click.option("--output_dir", type=str, default='infer_out', required=False)
def wav2wav(wav, config, keyshift, output_dir):
    set_hparams(config=config, exp_name='vocoder')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder = VOCODERS[hparams['vocoder']]()
    vocoder.to_device(device)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(wav):
        wav_files = [os.path.join(wav, f) for f in os.listdir(wav) if f.endswith('.wav')]
    else:
        wav_files = [wav]
    for wav_file in wav_files:
        # wav2spec
        wave, mel = vocoder.wav2spec(wav_file, hparams=hparams, keyshift=keyshift)
        # spec2wav
        f0, _ = get_pitch(wave, mel, hparams)
        if keyshift != 0:
            f0 = shift_pitch(f0, keyshift)
        res = vocoder.spec2wav(mel, f0=f0)
        title = os.path.basename(wav_file).split('.')[0]
        save_wav(res, os.path.join(output_dir, f"{title}.wav"), hparams['audio_sample_rate'])

if __name__ == "__main__":
    main()  