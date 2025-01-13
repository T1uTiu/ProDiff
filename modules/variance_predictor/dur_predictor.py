import torch

from modules.commons.common_layers import Embedding, Linear
from modules.fastspeech.tts_modules import FastspeechEncoder, DurationPredictor


class DurPredictor(torch.nn.Module):
    def __init__(self, vocab_size, hparams):
        super(DurPredictor, self).__init__()
        self.encoder = FastspeechEncoder(
            vocab_size=vocab_size, 
            hidden_size=hparams['hidden_size'], 
            num_layers=hparams['enc_layers'], 
            kernel_size=hparams['enc_ffn_kernel_size'], 
            num_heads=hparams['num_heads']
        )
        # duration
        dur_hparams = hparams['dur_prediction_args']
        self.onset_embed = Embedding(2, hparams['hidden_size'])
        self.word_dur_embed = Linear(1, hparams['hidden_size'])
        self.dur_pred = DurationPredictor(
            in_dims=hparams['hidden_size'],
            n_layers=dur_hparams['num_layers'],
            n_chans=dur_hparams['hidden_size'],
            dropout_rate=dur_hparams['dropout'],
            kernel_size=dur_hparams['kernel_size'],
            offset=dur_hparams['log_offset'],
            dur_loss_type=dur_hparams['loss_type'],
        )
    
    def forward(self, txt_tokens, onset, word_dur, infer=True):
        extra_embed = self.onset_embed(onset)
        extra_embed += self.word_dur_embed(word_dur[:, :, None])
        encoder_out = self.encoder(txt_tokens, extra_embed)
        ph_dur_pred = self.dur_pred(encoder_out, x_masks=txt_tokens == 0, infer=infer)
        return ph_dur_pred
