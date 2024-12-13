#########
# world
##########
import librosa
import numpy as np
import torch

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def norm_f0(f0, uv, hparams):
    if uv is None:
        uv = f0 == 0
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = (f0 - hparams['f0_mean']) / hparams['f0_std']
    if hparams['pitch_norm'] == 'log':
        f0 = torch.log2(f0+uv) if is_torch else np.log2(f0+uv)
    f0[uv] = -np.inf
    return f0

def interp_f0(f0, uv, hparams):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv, hparams=hparams)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None, hparams=hparams), uv

def norm_interp_f0(f0, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, hparams)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    uv = torch.FloatTensor(uv)
    f0 = torch.FloatTensor(f0)
    if is_torch:
        f0 = f0.to(device)
    return f0, uv


def denorm_f0(f0, uv, hparams, pitch_padding=None, min=None, max=None):
    if hparams['pitch_norm'] == 'standard':
        f0 = f0 * hparams['f0_std'] + hparams['f0_mean']
    if hparams['pitch_norm'] == 'log':
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0

def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0: # 插值后的长度大于align_length
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp

def setuv_f0(f0, ph, durations, phone_uv_set):
    time_start = 0
    for i, phone in enumerate(ph):
        if phone in phone_uv_set:
            f0[time_start:min(time_start + durations[i], len(f0))] = 0.0
        time_start += durations[i]
    return f0

def shift_pitch(f0, n):
    return f0 * (2** (n / 12))

def random_continuous_masks(*shape: int, dim: int, device = 'cpu'):
    start, end = torch.sort(
        torch.randint(
            low=0, high=shape[dim] + 1, size=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1))), device=device
        ).expand(*((-1,) * (dim + 1)), *shape[dim + 1:]), dim=dim
    )[0].split(1, dim=dim)
    idx = torch.arange(
        0, shape[dim], dtype=torch.long, device=device
    ).reshape(*((1,) * dim), shape[dim], *((1,) * (len(shape) - dim - 1)))
    masks = (idx >= start) & (idx < end)
    return masks