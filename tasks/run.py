import os, sys
sys.path.append(os.getcwd())

import importlib
from utils.hparams import set_hparams, hparams



def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1]) # 包名
    cls_name = hparams["task_cls"].split(".")[-1] # 类名
    task_cls = getattr(importlib.import_module(pkg), cls_name) # 通过包名和类名获取类
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
