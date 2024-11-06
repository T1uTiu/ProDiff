import os, sys
sys.path.append(os.getcwd())

import json
import click
import numpy as np
from inference.ProDiff_Acoustic import ProDiffInfer
from inference.ProDiff_Teacher_Acoustic import ProDiffTeacherInfer
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

@click.group()
def infer():
    pass

@infer.command()
@click.argument("proj", type=str)
@click.option("--config", type=str)
@click.option("--exp_name", type=str)
@click.option("--spk_name", type=str)
def teacher(proj, config, exp_name, spk_name):
    set_hparams(config=config, exp_name=exp_name, spk_name=spk_name)
    hparams['title'] = proj.split('/')[-1].split('.')[0]
    with open('dictionaries/phone_uv_set.json', 'r', encoding='utf-8') as f:
        phone_uv_set = json.load(f)
        hparams['phone_uv_set'] = set(phone_uv_set)

    with open(proj, 'r', encoding='utf-8') as f:
        project = json.load(f)

    inferer = ProDiffTeacherInfer(hparams)
    os.makedirs('infer_out', exist_ok=True)

    result = []
    total_length = 0
    
    for segment in project:
        out = inferer.infer_once(segment)
        offset = int(segment.get('offset', 0) * hparams["audio_sample_rate"])
        out = np.concatenate([np.zeros(max(offset-total_length, 0)), out])
        total_length += len(out)
        result.append(out)
    
    save_wav(np.concatenate(result), f'infer_out/{hparams["title"]}【{hparams["exp_name"]}】.wav', hparams['audio_sample_rate'])

@infer.command()
@click.argument("proj", type=str)
@click.option("--config", type=str)
@click.option("--exp_name", type=str)
@click.option("--spk_name", type=str)
def student(proj, config, exp_name, spk_name):
    set_hparams(config=config, exp_name=exp_name, spk_name=spk_name)
    
    with open('dictionaries/phone_uv_set.json', 'r', encoding='utf-8') as f:
        phone_uv_set = json.load(f)
        hparams['phone_uv_set'] = set(phone_uv_set)

    with open(proj, 'r', encoding='utf-8') as f:
        project = json.load(f)

    inferer = ProDiffInfer(hparams)
    os.makedirs('infer_out', exist_ok=True)
    
    result = []
    total_length = 0
    
    for segment in project:
        out = inferer.infer_once(segment)
        offset = int(segment.get('offset', 0) * hparams["audio_sample_rate"])
        out = np.concatenate([np.zeros(max(offset-total_length, 0)), out])
        total_length += len(out)
        result.append(out)
    
    save_wav(np.concatenate(result), f'infer_out/{hparams["title"]}【{hparams["exp_name"]}】.wav', hparams['audio_sample_rate'])

if __name__ == '__main__':
    infer()