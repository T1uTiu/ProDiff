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
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.dur_embed = Linear(1, self.hidden_size)
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        self.timestep = hparams["hop_size"] / hparams["audio_sample_rate"]

        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], self.hidden_size)

        if hparams['use_lang_id']:
            self.lang_embed = Embedding(hparams['num_lang'], self.hidden_size)

        self.f0_embed_type = hparams.get('f0_embed_type', 'discrete')
        if hparams['use_pitch_embed']:
            if self.f0_embed_type == 'discrete':
                self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            else:
                self.pitch_embed = Linear(1, self.hidden_size)
    
    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, mel2ph=None, f0=None, spk_embed_id=None, **kwargs):
        ret = {}

        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        dur_embed = self.dur_embed(dur[:, :, None])

        if hparams['use_lang_id']:
            lang_embed = self.lang_embed(kwargs.get('language'))
            extra_embed = dur_embed + lang_embed
        else:
            extra_embed = dur_embed

        encoder_out = self.encoder(txt_tokens, extra_embed)  # [B, T, C]

        ret['mel2ph'] = mel2ph
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch embed
        decoder_inp += self.add_pitch(f0, ret)

        # add spk embed
        if hparams['use_spk_id']:
            decoder_inp += self.add_spk_embed(spk_embed_id, kwargs.get('spk_mix_embed'))

        ret['decoder_inp'] = decoder_inp = decoder_inp * tgt_nonpadding

        return ret

    def add_spk_embed(self, spk_embed_id, spk_mix_embed):
        if spk_mix_embed is not None:
            spk_embed = spk_mix_embed
        else:
            spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
        return spk_embed

    def add_pitch(self, f0:torch.Tensor, ret):
        ret['f0_denorm'] = f0
        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)  # start from 0
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, : , None])
        return pitch_embed

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        f0 = cwt2f0(cwt_spec, mean, std, hparams['cwt_scales'])
        f0 = torch.cat(
            [f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None, hparams)
        return f0_norm

    def out2mel(self, out):
        return out

    @staticmethod
    def mel_norm(x):
        return (x + 5.5) / (6.3 / 2) - 1

    @staticmethod
    def mel_denorm(x):
        return (x + 1) * (6.3 / 2) - 5.5
