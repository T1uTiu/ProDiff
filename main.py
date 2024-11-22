import json
import os
from typing import Dict
import click
import numpy as np
import torch

from inference.ProDiff_Acoustic import ProDiffInfer
from inference.ProDiff_Teacher_Acoustic import ProDiffTeacherInfer
from inference.base_tts_infer import BaseTTSInfer
from preprocess.base_binarizer import BaseBinarizer
from train.prodiff_task import ProDiffTask
from train.prodiff_teacher_task import ProDiffTeacherTask
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
    binarizer = BaseBinarizer()
    binarizer.process()

trainer_map: Dict[str, BaseTask] = {
    "teacher": ProDiffTeacherTask,
    "student": ProDiffTask,
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

inferer_map: Dict[str, BaseTTSInfer] = {
    "teacher": ProDiffTeacherInfer,
    "student": ProDiffInfer
}

@main.command()
@click.argument("inferer", type=str)
@click.argument("proj", type=str)
@click.option("--config", type=str)
@click.option("--exp_name", type=str)
@click.option("--spk_name", type=str)
def infer(inferer, proj, config, exp_name, spk_name):
    assert inferer in inferer_map, f"Invalid inferer: {inferer}, use one of {list(inferer_map.keys())}"

    set_hparams(config=config, exp_name=exp_name, spk_name=spk_name)
    
    with open('dictionaries/phone_uv_set.json', 'r', encoding='utf-8') as f:
        phone_uv_set = json.load(f)
        hparams['phone_uv_set'] = set(phone_uv_set)

    with open(proj, 'r', encoding='utf-8') as f:
        project = json.load(f)

    inferer_instance = inferer_map[inferer](hparams)
    os.makedirs('infer_out', exist_ok=True)
    
    result = []
    total_length = 0
    
    for segment in project:
        out = inferer_instance.infer_once(segment)
        offset = int(segment.get('offset', 0) * hparams["audio_sample_rate"])
        out = np.concatenate([np.zeros(max(offset-total_length, 0)), out])
        total_length += len(out)
        result.append(out)

    title = proj.split('/')[-1].split('.')[0]
    save_wav(np.concatenate(result), f'infer_out/{title}【{hparams["exp_name"]}】.wav', hparams['audio_sample_rate'])

@main.group()
def vocode():
    pass

@vocode.command()
@click.argument("wav", type=str)
@click.option("--config", type=str, required=True)
@click.option("--keyshift", type=int, default=0)
def wav2wav(wav, config, keyshift):
    set_hparams(config=config, exp_name='vocoder')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder = VOCODERS[hparams['vocoder']]()
    vocoder.to_device(device)
    # wav2spec
    wave, mel = vocoder.wav2spec(wav, hparams=hparams, keyshift=keyshift)
    # spec2wav
    f0, _ = get_pitch(wave, mel, hparams)
    if keyshift != 0:
        f0 = shift_pitch(f0, keyshift)
    res = vocoder.spec2wav(mel, f0=f0)
    title = os.path.basename(wav).split('.')[0]
    save_wav(res, f'infer_out/{title}.wav', hparams['audio_sample_rate'])

@vocode.command()
@click.argument("spec", type=str)
@click.option("--config", type=str, required=True)
def spec2wav(spec, config):
    pass

if __name__ == "__main__":
    main()  