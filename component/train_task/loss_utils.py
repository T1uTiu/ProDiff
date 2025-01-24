import torch
from torch import Tensor, nn
from torch.functional import F

from modules.commons.ssim import ssim

def weights_nonzero_speech(target):
    # target : [B, T, mel_bin]
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

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

class RectifiedFlowLoss(nn.Module):
    def __init__(self, loss_type, log_norm=True):
        super().__init__()
        self.loss_type = loss_type
        self.log_norm = log_norm
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type in ("l2", "mse"):
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(v_pred, v_gt, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return v_pred * non_padding, v_gt * non_padding
        else:
            return v_pred, v_gt

    @staticmethod
    def get_weights(t):
        eps = 1e-7
        t = t.float()
        t = torch.clip(t, 0 + eps, 1 - eps)
        weights = 0.398942 / t / (1 - t) * torch.exp(
            -0.5 * torch.log(t / (1 - t)) ** 2
        ) + eps
        return weights[:, None, None, None]

    def _forward(self, v_pred, v_gt, t=None):
        if self.log_norm:
            return self.get_weights(t) * self.loss(v_pred, v_gt)
        else:
            return self.loss(v_pred, v_gt)

    def forward(self, v_pred: Tensor, v_gt: Tensor, t: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param v_pred: [B, 1, M, T]
        :param v_gt: [B, 1, M, T]
        :param t: [B,]
        :param non_padding: [B, T, M]
        """
        v_pred, v_gt = self._mask_non_padding(v_pred, v_gt, non_padding)
        return self._forward(v_pred, v_gt, t=t).mean()

class PitchLoss(RectifiedFlowLoss):
    pass

class MelLoss(RectifiedFlowLoss):
    pass