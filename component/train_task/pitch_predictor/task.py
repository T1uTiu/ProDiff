import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_mel_loss
from component.train_task.pitch_predictor.dataset import PitchPredictorDataset
from modules.variance_predictor.pitch_predictor import PitchPredictor
import utils
from utils.plot import spec_to_figure


class PitchPredictorTask(BaseTask):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.build_phone_encoder()
        self.f0_prediction_args = hparams['f0_prediction_args']
        self.f0_repeat = [1, 1, self.f0_prediction_args['repeat_bins']]
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_and_lambda = {}
        for l in mel_losses:
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print("| Pitch losses:", self.loss_and_lambda)

    def get_dataset_cls(self):
        return PitchPredictorDataset
    
    def build_model(self):
        self.model = PitchPredictor(self.ph_encoder, self.hparams)

    def run_model(self, sample, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        base_f0 = sample["base_f0"]
        spk_id = sample.get("spk_id", None)
        # 模型输出
        f0_pred = self.model(txt_tokens, mel2ph, base_f0, f0, spk_id=spk_id, infer=infer)

        losses = {}
        add_mel_loss(f0_pred, f0.unsqueeze(-1).repeat(*self.f0_repeat), losses, loss_and_lambda=self.loss_and_lambda)
        if not return_output:
            return losses
        else:
            return losses, f0_pred
    
    def validation_step(self, sample, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], model_out = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < self.hparams['num_valid_plots']:
            _, delta_f0_pred = self.run_model(sample, return_output=True, infer=True)
            f0_pred = delta_f0_pred + sample["base_f0"]
            self.plot_spec(batch_idx, sample["f0"], f0_pred)
        return outputs
    
    def plot_f0_spec(self, batch_idx, spec, spec_out, name=None):
        spec = spec.unsqueeze(-1).repeat(*self.f0_repeat)
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'f0_{batch_idx}' if name is None else name
        vmin = self.f0_prediction_args['spec_min']
        vmax = self.f0_prediction_args['spec_max']
        self.logger.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)