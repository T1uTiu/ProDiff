import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


def spec_f0_to_figure(f0_tgt, f0_pred):
    if isinstance(f0_tgt, torch.Tensor):
        f0_tgt = f0_tgt.cpu().numpy()
    if isinstance(f0_pred, torch.Tensor):
        f0_pred = f0_pred.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(f0_tgt, color='r', label='gt')
    plt.plot(f0_pred, color='b', label='pred')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt):
    dur_gt = dur_gt.long().cpu().numpy()
    dur_pred = dur_pred.long().cpu().numpy()
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dur_gt)):
        shift = (i % 8) + 1
        plt.text(dur_gt[i], shift, txt[i])
        plt.text(dur_pred[i], 10 + shift, txt[i])
        plt.vlines(dur_gt[i], 0, 10, colors='b')  # blue is gt
        plt.vlines(dur_pred[i], 10, 20, colors='r')  # red is pred
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig
