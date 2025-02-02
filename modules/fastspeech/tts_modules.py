import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.common_layers import Embedding, SinusoidalPositionalEmbedding, Linear, EncSALayer, DecSALayer, BatchNorm1dTBC
from utils.text_encoder import PAD_ID

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size, num_heads=2, norm='ln', padding="SAME", act='gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm, act=act
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


######################
# fastspeech modules
######################
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, in_dims, n_layers=2, n_chans=384, kernel_size=3,
                 dropout_rate=0.1, offset=1.0, dur_loss_type='mse'):
        """Initialize duration predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv.append(torch.nn.Sequential(
                torch.nn.Identity(),  # this is a placeholder for ConstantPad1d which is now merged into Conv1d
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=kernel_size // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            ))

        self.loss_type = dur_loss_type
        if self.loss_type in ['mse', 'huber']:
            self.out_dims = 1
        else:
            raise NotImplementedError()
        self.linear = torch.nn.Linear(n_chans, self.out_dims)

    def out2dur(self, xs):
        if self.loss_type in ['mse', 'huber']:
            # NOTE: calculate loss in log domain
            dur = xs.squeeze(-1).exp() - self.offset  # (B, Tmax)
        else:
            raise NotImplementedError()
        return dur

    def forward(self, xs, x_masks=None, infer=True):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (BoolTensor, optional): Batch of masks indicating padded part (B, Tmax).
            infer (bool): Whether inference
        Returns:
            (train) FloatTensor, (infer) LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        masks = 1 - x_masks.float()
        masks_ = masks[:, None, :]
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * masks_
        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * masks[:, :, None]  # (B, T, C)

        dur_pred = self.out2dur(xs)
        if infer:
            dur_pred = dur_pred.clamp(min=0.)  # avoid negative value
        return dur_pred


class LengthRegulator(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha > 0
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class EnergyPredictor(PitchPredictor):
    pass


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


class FFTBlocks(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=0.1, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastspeechEncoder(FFTBlocks):
    def __init__(self, vocab_size, hidden_size, num_layers, kernel_size, dropout=0.1, num_heads=2,  rel_pos=False):
        super().__init__(hidden_size, num_layers, kernel_size, dropout=dropout, num_heads=num_heads,
                         use_pos_embed=False)  # use_pos_embed_alpha for compatibility
        self.embed_tokens = Embedding(vocab_size, hidden_size, PAD_ID)
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

    def forward(self, txt_tokens, extra_embed=None):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens, extra_embed=extra_embed, padding_mask=encoder_padding_mask)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

    def forward_embedding(self, txt_tokens, extra_embed=None, padding_mask=None):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        if extra_embed is not None:
            x = x + extra_embed
        if self.rel_pos:
            x = self.embed_positions(x)
        else:
            positions = self.embed_positions(~padding_mask)
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class NoteEncoder(FFTBlocks):
    def __init__(self, hidden_size, num_layers, kernel_size, dropout=0.1, num_heads=2, rel_pos=False):
        super().__init__(hidden_size, num_layers, kernel_size, dropout=dropout, num_heads=num_heads,
                         use_pos_embed=False)  # use_pos_embed_alpha for compatibility
        self.note_midi_embed = Linear(1, hidden_size)
        self.note_dur_embed = Linear(1, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

    def forward(self, note_midi, note_rest, note_dur):
        encoder_padding_mask = note_midi < 0
        x = self.forward_embedding(note_midi, note_rest, note_dur)  # [B, T, H]
        x = super(NoteEncoder, self).forward(x, encoder_padding_mask)
        return x

    def forward_embedding(self, note_midi, note_rest, note_dur):
        # embed tokens and positions
        x = self.embed_scale * self.note_midi_embed(note_midi[:, :, None]) * ~note_rest[:, :, None]
        x += self.note_dur_embed(note_dur[:, :, None])
        if self.rel_pos:
            x = self.embed_positions(x)
        else:
            padding_mask = note_midi < 0
            positions = self.embed_positions(~padding_mask)
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FastspeechDecoder(FFTBlocks):
    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)

