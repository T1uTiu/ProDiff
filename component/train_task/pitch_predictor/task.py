import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_mel_loss
from component.train_task.pitch_predictor.dataset import PitchPredictorDataset
from modules.variance_predictor.pitch_predictor import PitchPredictor
import utils
from utils.plot import spec_f0_to_figure


class PitchPredictorTask(BaseTask):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.f0_prediction_args = hparams['f0_prediction_args']
        self.f0_repeat = [1, 1, self.f0_prediction_args['repeat_bins']]
        pitch_losses = self.f0_prediction_args['loss_type'].split("|")
        self.loss_and_lambda = {}
        for l in pitch_losses:
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
        self.model = PitchPredictor(self.hparams)

    def run_model(self, sample, return_output=False, infer=False):
        note_midi = sample["note_midi"] 
        note_rest = sample["note_rest"]
        mel2note = sample["mel2note"]
        f0 = sample["pitch"]
        base_f0 = sample["base_pitch"]
        spk_id = sample.get("spk_id", None)
        # 模型输出
        f0_pred = self.model(note_midi, note_rest, mel2note, 
                             base_f0, f0=f0, 
                             spk_id=spk_id, infer=infer)
        if infer:
            return f0_pred
        losses = {}
        f0_gt = f0 - base_f0
        f0_gt = f0_gt.unsqueeze(-1).repeat(*self.f0_repeat)
        add_mel_loss(f0_pred, f0_gt, losses, loss_and_lambda=self.loss_and_lambda)
        if not return_output:
            return losses
        else:
            return losses, f0_pred
    
    def validation_step(self, sample, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], _ = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < self.hparams['num_valid_plots']:
            f0_pred = self.run_model(sample, return_output=True, infer=True)
            f0_gt = sample["pitch"]
            f0_pred += sample["base_pitch"]
            self.plot_f0_spec(batch_idx, f0_gt, f0_pred)
        return outputs
    
    def plot_f0_spec(self, batch_idx, f0_tgt, f0_pred):
        name = f'f0_{batch_idx}'
        self.logger.add_figure(name, spec_f0_to_figure(f0_tgt[0], f0_pred[0]), self.global_step)