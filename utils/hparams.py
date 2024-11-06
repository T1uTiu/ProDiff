import argparse
import os
import subprocess

import yaml

global_print_hparams = True
task_cls_mapping = {
    "ProDiff_Teacher": "modules.ProDiff.task.ProDiff_teacher_task.ProDiff_teacher_Task",
    "ProDiff": "modules.ProDiff.task.ProDiff_task.ProDiff_Task",
}
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', spk_name='', hparams_str='', print_hparams=True, global_hparams=True):
    if config == '' and exp_name == '':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', type=str, default='configs/config_base.yaml',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name') # 模型名称
        parser.add_argument('--hparams', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--remove', action='store_true', help='remove old ckpt')
        parser.add_argument('--debug', action='store_true', help='debug')
        parser.add_argument("--proj", type=str) # project name
        parser.add_argument("--task_cls", type=str, default="ProDiff") # teacher/student
        parser.add_argument("--spk_name", type=str, default="")
        args, unknown = parser.parse_known_args()
    else:
        args = Args(config=config, exp_name=exp_name, task_cls="ProDiff", hparams=hparams_str,
                    infer=False, validate=False, reset=True, remove=False, debug=False, spk_name=spk_name)
    global hparams
    assert args.config != '' and args.exp_name != ''

    config_chains = []
    loaded_config = set()

    def load_config(config_fn):  # 深度优先搜索
        if not os.path.exists(config_fn):
            return {}
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        loaded_config.add(config_fn)
        if 'base_config' in hparams_:
            ret_hparams = {}
            if not isinstance(hparams_['base_config'], list):
                hparams_['base_config'] = [hparams_['base_config']]
            for c in hparams_['base_config']:
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                if c not in loaded_config:
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams

    args_work_dir = f'checkpoints/{args.exp_name}'
    ckpt_config_path = f'{args_work_dir}/config.yaml'
    
    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(args.config))
    if not args.reset:
        saved_hparams = {}
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                saved_hparams.update(yaml.safe_load(f))
        hparams_.update(saved_hparams)
    hparams_['work_dir'] = args_work_dir

    # --hparams="a=1,b.c=2,d=[1 1 1]"
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            for k_ in k.split(".")[:-1]:
                config_node = config_node[k_]
            k = k.split(".")[-1]
            if v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                config_node[k] = eval(v)
            else:
                config_node[k] = type(config_node[k])(v)
    if args_work_dir != '' and args.remove:
        answer = input("REMOVE old checkpoint? Y/N [Default: N]: ")
        if answer.lower() == "y":
            subprocess.check_call(f'rm -rf {args_work_dir}', shell=True)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)

    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name
    hparams_['spk_name'] = args.spk_name
    assert args.task_cls in task_cls_mapping, f"task_cls should be in {task_cls_mapping.keys()}"
    hparams_["task_cls"] = task_cls_mapping[args.task_cls]
        
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_
