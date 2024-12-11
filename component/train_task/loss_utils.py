import torch
from torch.functional import F

from modules.commons.ssim import ssim

def weights_nonzero_speech(target):
    # target : [B, T, mel_bin]
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

def l1_loss(self, decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    l1_loss = F.l1_loss(decoder_output, target, reduction='none')
    weights = weights_nonzero_speech(target)
    l1_loss = (l1_loss * weights).sum() / weights.sum()
    return l1_loss

def mse_loss(self, decoder_output, target):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    mse_loss = F.mse_loss(decoder_output, target, reduction='none')
    weights = weights_nonzero_speech(target)
    mse_loss = (mse_loss * weights).sum() / weights.sum()
    return mse_loss

def ssim_loss(self, decoder_output, target, bias=6.0):
    # decoder_output : B x T x n_mel
    # target : B x T x n_mel
    assert decoder_output.shape == target.shape
    weights = weights_nonzero_speech(target)
    decoder_output = decoder_output[:, None] + bias
    target = target[:, None] + bias
    ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
    ssim_loss = (ssim_loss * weights).sum() / weights.sum()
    return ssim_loss

def add_mel_loss(mel_out, target, losses, loss_and_lambda, postfix=''):
    nonpadding = target.abs().sum(-1).ne(0).float()
    for loss_name, lbd in loss_and_lambda.items():
        if 'l1' == loss_name:
            l = l1_loss(mel_out, target)
        elif 'mse' == loss_name:
            l = mse_loss(mel_out, target)
        elif 'ssim' == loss_name:
            l = ssim_loss(mel_out, target)
        else:
            raise NotImplementedError()
        losses[f'{loss_name}{postfix}'] = l * lbd

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
