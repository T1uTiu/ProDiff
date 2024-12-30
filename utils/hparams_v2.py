import os
import yaml

hparams = {}

def set_hparams(config_fn=None, exp_name=None, task=None, global_hparams=True):
    global hparams
    if config_fn is None or not os.path.exists(config_fn):
        assert task is not None, "You should at least provide config or task"
        config_fn = f"checkpoints"
        if exp_name is not None:
            config_fn = os.path.join(config_fn, exp_name)
        config_fn = os.path.join(config_fn, task, "config.yaml")
    assert os.path.exists(config_fn), f"Config file not found: {config_fn}"
    
    with open(config_fn) as f:
        _hparams = yaml.safe_load(f)

    _hparams["task"] = task
    if exp_name is not None:
        _hparams['exp_name'] = exp_name
        _hparams['work_dir'] = os.path.join("checkpoints", exp_name, task)
    else:
        _hparams['work_dir'] = os.path.join("checkpoints", task)
    os.makedirs(_hparams['work_dir'], exist_ok=True)
    
    if global_hparams:
        hparams = _hparams
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(_hparams.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")

    return _hparams