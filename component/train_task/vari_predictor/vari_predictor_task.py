from component.train_task.vari_predictor.dataset import VariPredictorDataset
from modules.variance_predictor.vari_predictor import VariPredictor
import utils
from utils.hparams import hparams
from tasks.tts.fs2 import FastSpeech2Task
from utils.plot import dur_to_figure
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder



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
        self.add_dur_loss(dur_pred, dur_tgt, onset, losses)
        if not return_output:
            return losses
        else:
            return losses, dur_pred

    def validation_step(self, sample: dict, batch_idx):
        outputs = {}
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        word_dur = sample["word_dur"]  # [B, T_w]
        onset = sample["onset"]  # [B, T_ph]
        target = sample["ph_dur"]  # [B, T_ph]

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, onset, word_dur, infer=True)
            self.plot_dur(batch_idx, sample, model_out)
        return outputs

    def plot_dur(self, batch_idx, sample, dur_pred):
        dur_tgt = sample["ph_dur"]
        ph_text = self.phone_encoder.decode(sample["ph_seq"][0].cpu().numpy()).split()
        # self.logger.add_figure(
        #     f"dur_{batch_idx}", dur_to_figure(dur_tgt, dur_pred, ph_text), self.global_step
        # )
        print(f"ph_text: {ph_text[0]}\ndur_tgt: {dur_tgt[0]}\ndur_pred: {dur_pred[0]}")

