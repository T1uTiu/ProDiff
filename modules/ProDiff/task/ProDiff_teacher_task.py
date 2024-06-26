import torch

import utils
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from usr.diff.net import DiffNet
from tasks.tts.fs2 import FastSpeech2Task
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from utils.pitch_utils import denorm_f0
from tasks.tts.fs2_utils import FastSpeechDataset

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class ProDiff_teacher_Task(FastSpeech2Task):
    def __init__(self):
        super(ProDiff_teacher_Task, self).__init__()
        self.dataset_cls = FastSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def build_model(self):
        self.build_tts_model()
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def build_tts_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )


    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        spk_embed_id = sample.get('spk_ids') if hparams['use_spk_id'] else None
        # 模型输出
        output = model(txt_tokens, mel2ph=mel2ph, spk_embed_id=spk_embed_id,
                       ref_mels=target, f0=f0, infer=infer)

        losses = {}
        self.add_mel_loss(output['mel_out'], target, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        spk_embed_id = sample.get('spk_ids') if hparams['use_spk_id'] else None
        mel2ph = sample['mel2ph']
        f0 = sample['f0']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, mel2ph=mel2ph, spk_embed_id=spk_embed_id, f0=f0, ref_mels=None, infer=True)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
        return outputs

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

