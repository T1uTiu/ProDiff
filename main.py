import os
import click
import torch

from handler.binarize import BinarizeHandler
from handler.infer import InferHandler
from component.train_task import SVSTask, DurPredictorTask, PitchPredictorTask
from handler.train.handler import TrainHandler
from utils.data_gen_utils import get_pitch
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams
from utils.hparams_v2 import set_hparams as set_hparams_v2
from utils.pitch_utils import shift_pitch
from component.vocoder.base_vocoder import get_vocoder_cls

@click.group()
def main():
    pass

@main.command()
@click.argument("task", type=str)
@click.option("--config", type=str, required=True)
@click.option("--exp_name", type=str, required=True)
def binarize(task, config, exp_name):
    exp_name = f"{exp_name}_{task}"
    hparams = set_hparams_v2(config=config, exp_name=exp_name, task=task)
    BinarizeHandler(hparams=hparams).handle()

train_task_map = {
    "svs": SVSTask,
    "dur": DurPredictorTask,
    "pitch": PitchPredictorTask
}

@main.command()
@click.argument("train_task", type=str)
@click.option("--config", type=str, required=True)
@click.option("--exp_name", type=str, required=True)
def train(train_task, config, exp_name):
    assert train_task in train_task_map, f"Invalid train task: {train_task}, use one of {list(train_task_map.keys())}"
    exp_name = f"{exp_name}_{train_task}"
    set_hparams(config=config, exp_name=exp_name)
    hparams["task"] = train_task
    TrainHandler(hparams=hparams).handle(train_task_map[train_task])


@main.command()
@click.argument("proj", type=str)
@click.option("--exp_name", type=str, required=True)
@click.option("--spk_name", type=str, required=True)
@click.option("--lang", type=str, default='zh')
@click.option("--keyshift", type=int, default=0)
@click.option("--gender", type=int, default=0)
@click.option("--pred_dur", is_flag=True)
@click.option("--pred_pitch", type=str, default="")
def infer(proj, exp_name, spk_name, lang, keyshift, gender, pred_dur, pred_pitch):
    InferHandler(exp_name=exp_name, pred_dur=pred_dur, pred_pitch=pred_pitch).handle(None, proj, spk_name, lang, keyshift, gender)

@main.group()
def vocode():
    pass

@vocode.command()
@click.argument("wav", type=str)
@click.option("--config", type=str, required=True)
@click.option("--keyshift", type=int, default=0, required=False)
@click.option("--output_dir", type=str, default='infer_out', required=False)
def wav2wav(wav, config, keyshift, output_dir):
    hparams = set_hparams_v2(config=config, task='vocoder', make_work_dir=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder = get_vocoder_cls(hparams["vocoder"])()
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
        f0, _ = get_pitch(wave, mel, hparams["hop_size"], hparams["audio_sample_rate"])
        if keyshift != 0:
            f0 = shift_pitch(f0, keyshift)
        res = vocoder.spec2wav(mel, f0=f0)
        title = os.path.basename(wav_file).split('.')[0]
        save_wav(res, os.path.join(output_dir, f"{title}.wav"), hparams['audio_sample_rate'])

if __name__ == "__main__":
    main()  