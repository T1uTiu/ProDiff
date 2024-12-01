import torch

from modules.fastspeech.tts_modules import FastspeechEncoder, LayerNorm


class DurationPredictor(torch.nn.Module):
    def __init__(self, in_dims, n_layers=2, n_chans=384, kernel_size=3,
                 dropout_rate=0.1, offset=1.0, dur_loss_type='mse'):
        super(DurationPredictor, self).__init__()
        self.encoder = FastspeechEncoder()
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