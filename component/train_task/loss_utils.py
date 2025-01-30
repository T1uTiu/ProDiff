from typing import Dict
import torch
from torch import nn
from torch.functional import F

from modules.commons.ssim import ssim

def ssim_loss(decoder_output, target, bias=6.0):
    """
    :param decoder_output: [B, F, M, T]
    :param target: [B, F, M, T]
    """
    decoder_output = decoder_output.transpose(-1, -2) + bias
    target = target.transpose(-1, -2) + bias
    ssim_loss = 1 - ssim(decoder_output, target)
    return ssim_loss

def add_sepc_loss_prodiff(
        pred_spec: torch.Tensor, gt_spec: torch.Tensor, non_padding: torch.Tensor,
        loss_type: Dict[str, float], 
        losses: Dict, name='spec'):
    """
    :param pred_spec: [B, F, M, T]
    :param gt_spec: [B, F, M, T]
    :param non_padding: [B, T, 1]
    """
    if non_padding is not None:
        non_padding = non_padding.transpose(1, 2).unsqueeze(1) # [B, 1, 1, T]
        pred_spec = pred_spec * non_padding
        gt_spec = gt_spec * non_padding
    for loss_name, lbd in loss_type.items():
        if 'l1' == loss_name:
            l = F.l1_loss(pred_spec, gt_spec)
        elif 'mse' == loss_name:
            l = F.mse_loss(pred_spec, gt_spec)
        elif 'ssim' == loss_name:
            l = ssim_loss(pred_spec, gt_spec)
        else:
            raise NotImplementedError()
        losses[f'{name}_{loss_name}'] = l * lbd

def add_spec_loss_reflow(
        pred_spec: torch.Tensor, gt_spec: torch.Tensor, t: torch.Tensor, non_padding: torch.Tensor,
        loss_type: str, log_norm: bool, 
        losses: Dict, name: str = "spec"
    ):
    """
    :param v_pred: [B, 1, M, T]
    :param v_gt: [B, 1, M, T]
    :param t: [B,]
    :param non_padding: [B, T, M]
    """
    if loss_type == "l1":
        loss_fn = nn.L1Loss()
    elif loss_type in ("l2", "mse"):
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError()
    if non_padding is not None:
        non_padding = non_padding.transpose(1, 2).unsqueeze(1)
        pred_spec = pred_spec * non_padding
        gt_spec = gt_spec * non_padding
    loss = loss_fn(pred_spec, gt_spec)
    if log_norm:
        eps = 1e-7
        t = t.float()
        t = torch.clip(t, 0 + eps, 1 - eps)
        weights = 0.398942 / t / (1 - t) * torch.exp(
            -0.5 * torch.log(t / (1 - t)) ** 2
        ) + eps
        loss = weights[:, None, None, None] * loss
    losses[name] = loss.mean()


def add_dur_loss(dur_pred, dur_tgt, onset, loss_type: str, log_offset, loss_and_lambda, losses):
    if loss_type == "mse":
        loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError()
    linear2log = lambda x: torch.log(x + log_offset)
    lambda_pdur = loss_and_lambda["ph"]
    lambda_wdur = loss_and_lambda["word"]
    lambda_sdur = loss_and_lambda["sentence"]
    # pdur loss
    pdur_loss = lambda_pdur * loss(linear2log(dur_pred), linear2log(dur_tgt))
    dur_pred = dur_pred.clamp(min=0.)
    # wdur loss
    ph2word = onset.cumsum(dim=1)
    shape = dur_pred.shape[0], ph2word.max()+1
    wdur_pred = dur_pred.new_zeros(*shape).scatter_add(
        1, ph2word, dur_pred
    )[:, 1:]
    wdur_tgt = dur_tgt.new_zeros(*shape).scatter_add(
        1, ph2word, dur_tgt
    )[:, 1:]
    wdur_loss = lambda_wdur * loss(linear2log(wdur_pred), linear2log(wdur_tgt))
    # sentence dur loss
    sdur_pred = dur_pred.sum(dim=1)
    sdur_tgt = dur_tgt.sum(dim=1)
    sdur_loss = lambda_sdur * loss(linear2log(sdur_pred), linear2log(sdur_tgt))
    losses["dur"] = pdur_loss + wdur_loss + sdur_loss
