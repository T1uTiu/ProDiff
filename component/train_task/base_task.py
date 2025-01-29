import json
from torch.utils.tensorboard import SummaryWriter
import traceback
from functools import wraps
import sys
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.utils.data
import utils
import logging
import os

from utils.common_schedulers import NoneSchedule, RSQRTSchedule
from utils.text_encoder import TokenTextEncoder

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


class BaseTask(nn.Module):
    def __init__(self, hparams):
        # dataset configs
        super(BaseTask, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hparams = hparams
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_valid_tokens = hparams['max_valid_tokens']
        if self.max_valid_tokens == -1:
            hparams['max_valid_tokens'] = self.max_valid_tokens = self.max_tokens
        self.max_valid_sentences = hparams['max_valid_sentences']
        if self.max_valid_sentences == -1:
            hparams['max_valid_sentences'] = self.max_valid_sentences = self.max_sentences
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_norm = hparams['clip_grad_norm']
        self.gradient_clip_val = hparams.get('clip_grad_value', 0)
        self.accumulate_grad_batches = hparams.get('accumulate_grad_batches', 1)
        self.model = None
        self.training_losses_meter = None
        self.logger: SummaryWriter = None
        self.data_dir = os.path.join(hparams["data_dir"], hparams["task"])
        self.dataset_cls = self.get_dataset_cls()

    ######################
    # build model, dataloaders, optimizer, scheduler, tensorboard
    ######################
    def build_model(self):
        raise NotImplementedError
    
    def get_dataset_cls(self):
        raise NotImplementedError

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix=self.hparams['train_set_name'], shuffle=True, hparams=self.hparams)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=self.hparams['endless_ds'])

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=self.hparams['test_set_name'], shuffle=False, hparams=self.hparams)
        return self.build_dataloader(test_dataset, False, self.max_valid_tokens, self.max_valid_sentences, batch_by_size=False)

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=self.hparams['valid_set_name'], shuffle=False, hparams=self.hparams)
        return self.build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences)

    def build_scheduler(self, optimizer):
        if self.hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer, self.hparams['lr'], self.hparams['warmup_updates'], self.hparams['hidden_size'])
        else:
            return NoneSchedule(optimizer, self.hparams['lr'])

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.hparams['lr'],
            betas=(self.hparams['optimizer_adam_beta1'], self.hparams['optimizer_adam_beta2']),
            weight_decay=self.hparams['weight_decay'])
        return optimizer

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, version, **kwargs):
        root_dir = os.path.join(save_dir, name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, "version_" + str(version))
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    def build_phone_encoder(self):
        ph_map_fn = os.path.join(self.data_dir, 'phone_set.json')
        with open(ph_map_fn, 'r') as f:
            ph_map = json.load(f)
        ph_list = list(sorted(set(ph_map.values())))
        self.ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov='SP')

    def build_phone_category_encoder(self):
        ph_category_list_fn = os.path.join(self.data_dir, 'ph_category_list.json')
        with open(ph_category_list_fn, 'r') as f:
            ph_category_list = json.load(f)
        self.ph_category_encoder = TokenTextEncoder(None, vocab_list=ph_category_list, replace_oov='SP')

    ######################
    # training
    ######################
    def on_train_start(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def run_model(self, sample: dict, return_output=False, infer=False):
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        """
        :param sample:
        :param batch_idx:
        :param optimizer_idx:
        :return: {'loss': torch.Tensor, 'progress_bar': dict, 'tb_log': dict}
        """
        log_outputs = self.run_model(sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample["nsamples"]
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        if optimizer_idx >= 0:
            log_outputs[f'lr_{optimizer_idx}'] = self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']

        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'tb_log': tb_log
        }

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // self.accumulate_grad_batches)

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        print(f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}")

    def on_train_end(self):
        pass

    ######################
    # validation
    ######################
    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: {"losses": {...}, "total_loss": float, ...} or (total loss: torch.Tensor, loss_log: dict)
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        all_losses_meter = {'total_loss': utils.AvgrageMeter()}
        for output in outputs:
            if len(output) == 0 or output is None:
                continue
            if isinstance(output, dict):
                assert 'losses' in output, 'Key "losses" should exist in validation output.'
                n = output.pop('nsamples', 1)
                losses = utils.tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                assert len(output) == 2, 'Validation output should only consist of two elements: (total_loss, losses)'
                n = 1
                total_loss, losses = output
                losses = utils.tensors_to_scalars(losses)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)
        loss_output = {k: round(v.avg, 4) for k, v in all_losses_meter.items()}
        print(f"| Valid results: {loss_output}")
        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    ######################
    # testing
    ######################
    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ######################
    # utils
    ######################
    def load_ckpt(self, ckpt_base_dir, current_model_name=None, model_name='model', force=True, strict=True):
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(self.__getattr__(current_model_name), ckpt_base_dir, current_model_name, force, strict)

    def on_keyboard_interrupt(self):
        pass
