from component.train_task.base_task import BaseTask
from component.train_task.dur_predictor.dataset import DurPredictorDataset
from component.train_task.loss_utils import add_dur_loss
from modules.variance_predictor.dur_predictor import DurPredictor
import utils



class DurPredictorTask(BaseTask):
    def __init__(self, hparams):
        super(DurPredictorTask, self).__init__(hparams=hparams)
        self.build_phone_encoder()
        loss_args = hparams["dur_prediction_args"]
        self.loss_type = loss_args["loss_type"]
        self.loss_log_offset = loss_args["log_offset"]
        self.loss_and_lambda = {
            "ph": loss_args["lambda_pdur_loss"],
            "word": loss_args["lambda_wdur_loss"],
            "sentence": loss_args["lambda_sdur_loss"],
        }
        print("| Dur losses:", self.loss_type, self.loss_and_lambda)

    def get_dataset_cls(self):
        return DurPredictorDataset

    def build_model(self):
        self.model = DurPredictor(len(self.ph_encoder), self.hparams)
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def run_model(self, sample, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        word_dur = sample["word_dur"]  # [B, T_w]
        onset = sample["onset"]  # [B, T_ph]
        dur_tgt = sample["ph_dur"]  # [B, T_ph]
        # 模型输出
        dur_pred = self.model(txt_tokens, onset, word_dur, infer=infer)

        losses = {}
        add_dur_loss(dur_pred, dur_tgt, onset, self.loss_type, self.loss_log_offset, self.loss_and_lambda, losses)
        if not return_output:
            return losses
        else:
            return losses, dur_pred

    def validation_step(self, sample: dict, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], model_out = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < self.hparams['num_valid_plots']:
            model_out = self.run_model(sample, return_output=True, infer=True)
            self.plot_dur(batch_idx, sample, model_out)
        return outputs

    def plot_dur(self, batch_idx, sample, dur_pred):
        dur_tgt = sample["ph_dur"]
        ph_text = self.ph_encoder.decode(sample["ph_seq"][0].cpu().numpy()).split()
        # self.logger.add_figure(
        #     f"dur_{batch_idx}", dur_to_figure(dur_tgt, dur_pred, ph_text), self.global_step
        # )
        print(f"ph_text: {ph_text[0]}\ndur_tgt: {dur_tgt[0]}\ndur_pred: {dur_pred[0]}")

