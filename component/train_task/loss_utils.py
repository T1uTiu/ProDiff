from typing import Dict
import torch
from torch import Tensor, nn
from torch.functional import F

from modules.commons.ssim import ssim

def weights_nonzero_speech(target):
    # target : [B, F, M, T]
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-2)
    return target.abs().sum(-2, keepdim=True).ne(0).float().repeat(1, 1, dim, 1)

def l1_loss(decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    l1_loss = F.l1_loss(decoder_output, target, reduction='none')
    weights = weights_nonzero_speech(target)
    l1_loss = (l1_loss * weights).sum() / weights.sum()
    return l1_loss

def mse_loss(decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    mse_loss = F.mse_loss(decoder_output, target, reduction='none')
    weights = weights_nonzero_speech(target)
    mse_loss = (mse_loss * weights).sum() / weights.sum()
    return mse_loss

def ssim_loss(decoder_output, target, bias=6.0):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    weights = weights_nonzero_speech(target)
    decoder_output = decoder_output + bias
    target = target + bias
    ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
    ssim_loss = (ssim_loss * weights).sum() / weights.sum()
    return ssim_loss

def add_sepc_loss_prodiff(
        pred_spec: torch.Tensor, gt_spec: torch.Tensor, non_padding: torch.Tensor,
        loss_type: Dict[str, float], 
        losses: Dict, name='spec'):
    """
    :param pred_spec: [B,  M, T]
    :param gt_spec: [B, M, T]
    """
    if non_padding is not None:
        non_padding = non_padding.transpose(1, 2).unsqueeze(1)
        pred_spec = pred_spec * non_padding
        gt_spec = gt_spec * non_padding
    for loss_name, lbd in loss_type.items():
        if 'l1' == loss_name:
            l = l1_loss(pred_spec, gt_spec)
        elif 'mse' == loss_name:
            l = mse_loss(pred_spec, gt_spec)
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
