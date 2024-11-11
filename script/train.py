import os, sys
sys.path.append(os.getcwd())

import click
from utils.hparams import set_hparams
from modules.ProDiff.task.ProDiff_teacher_task import ProDiff_teacher_Task
from modules.ProDiff.task.ProDiff_task import ProDiff_Task

trainer_map = {
    "teacher": ProDiff_teacher_Task,
    "student": ProDiff_Task,
}

@click.command()
@click.argument("trainer", type=str)
@click.option("--config", type=str, required=True)
@click.option("--exp_name", type=str, required=True)
def train(trainer, config, exp_name):
    assert trainer in trainer_map, f"Invalid trainer: {trainer}, use one of {list(trainer_map.keys())}"
    set_hparams(config=config, exp_name=exp_name)
    trainer_instance = trainer_map[trainer]
    trainer_instance.start()



if __name__ == '__main__':
    train()