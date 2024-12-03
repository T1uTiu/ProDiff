from component.train_task.dataset import ProDiffDataset, ProDiffDatasetBatchItem
from component.train_task.vari_predictor.dataset import VariPredictorDataset
from modules.ProDiff.prodiff_teacher import ProDiffTeacher
from modules.variance_predictor.vari_predictor import VariPredictor
import utils
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from usr.diff.net import DiffNet
from tasks.tts.fs2 import FastSpeech2Task
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class VariPredictorTask(FastSpeech2Task):
    def __init__(self):
        super(VariPredictorTask, self).__init__()
        self.dataset_cls = VariPredictorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def build_model(self):
        self.build_tts_model()
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def build_tts_model(self):
        self.model = VariPredictor(self.phone_encoder, hparams)


    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        word_dur = sample["word_dur"]  # [B, T_w]
        onset = sample["onset"]  # [B, T_ph]
        dur_tgt = sample["ph_dur"]  # [B, T_ph]
        # 模型输出
        dur_pred = model(txt_tokens, onset, word_dur, infer=infer)

        losses = {}
        self.add_dur_loss(dur_pred, dur_tgt, word_dur, onset, losses)
        if not return_output:
            return losses
        else:
            return losses, dur_pred

    def validation_step(self, sample: ProDiffDatasetBatchItem, batch_idx):
        outputs = {}
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        word_dur = sample["word_dur"]  # [B, T_w]
        onset = sample["onset"]  # [B, T_ph]
        target = sample["ph_dur"]  # [B, T_ph]

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample.nsamples
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, onset, word_dur, infer=True)
            self.plot_dur(batch_idx, target, model_out)
        return outputs

