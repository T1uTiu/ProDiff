from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FastspeechDecoder, FastspeechEncoder, mel2ph_to_dur
from utils.cwt import cwt2f0
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, norm_f0

FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}

FS_DECODERS = {
    'fft': lambda hp: FastspeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}


class FastSpeech2(nn.Module):
    def __init__(self, ph_encoder, hidden_size, enc_layers, enc_ffn_kernel_size, num_heads):
        super().__init__()
        padding_idx = ph_encoder.pad()
        self.dur_embed = Linear(1, hidden_size)
        self.encoder = FastspeechEncoder(ph_encoder, hidden_size, enc_layers, enc_ffn_kernel_size, num_heads)
        if hparams['use_spk_id']:
            self.spk_embed = Embedding(len(hparams['datasets']), hidden_size)
        if hparams['use_lang_id']:
            self.lang_embed = Embedding(len(hparams["dictionary"]), hidden_size, padding_idx)
        f0_embed_type = hparams.get('f0_embed_type', 'continuous')
        if f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, self.hidden_size, padding_idx)
        else:
            self.pitch_embed = Linear(1, self.hidden_size)

    def forward(self, txt_tokens, mel2ph, f0, lang_seq=None, spk_embed_id=None, spk_mix_embed=None, **kwargs):
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        extra_embed = self.dur_embed(dur[:, :, None])

        if hparams['use_lang_id']:
            lang_embed = self.lang_embed(lang_seq)
            extra_embed += lang_embed

        encoder_out = self.encoder(txt_tokens, extra_embed)  # [B, T, C]

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch embed
        decoder_inp += self.add_pitch(f0)

        # add spk embed
        if hparams['use_spk_id']:
            decoder_inp += self.add_spk_embed(spk_embed_id, spk_mix_embed)

        return decoder_inp * tgt_nonpadding

    def add_spk_embed(self, spk_embed_id, spk_mix_embed):
        assert not (spk_embed_id is None and spk_mix_embed is None)
        if spk_mix_embed is not None:
            spk_embed = spk_mix_embed
        else:
            spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
        return spk_embed

    def add_pitch(self, f0:torch.Tensor):
        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)  # start from 0
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, : , None])
        return pitch_embed

    @staticmethod
    def mel_norm(x):
        return (x + 5.5) / (6.3 / 2) - 1

    @staticmethod
    def mel_denorm(x):
        return (x + 1) * (6.3 / 2) - 5.5
