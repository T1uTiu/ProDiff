import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_mel_loss
from component.train_task.pitch_predictor.dataset import PitchPredictorDataset
from component.train_task.vari_predictor.dataset import VoicingPredictorDataset, BreathPredictorDataset
from modules.variance_predictor.pitch_predictor import PitchPredictor
from modules.variance_predictor.vari_predictor import VoicingPredictor, BreathPredictor
import utils
from utils.plot import spec_curse_to_figure


class VariPredictorTask(BaseTask):
    def __init__(self, hparams, vari_type):
        super().__init__(hparams=hparams)
        self.vari_type = vari_type
        self.vari_prediction_args = hparams[f'{vari_type}_prediction_args']
        self.vari_repeat = [1, 1, self.vari_prediction_args['repeat_bins']]
        vari_losses = self.vari_prediction_args['loss_type'].split("|")
        self.loss_and_lambda = {}
        for l in vari_losses:
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print(f"| {vari_type} losses:", self.loss_and_lambda)

    def run_model(self, sample, return_output=False, infer=False):
        tgt_vari = sample[self.vari_type]
        note_midi = sample["note_midi"] 
        note_rest = sample["note_rest"]
        mel2note = sample["mel2note"]
        f0 = sample["f0"]
        # 模型输出
        output = self.model(note_midi, note_rest, mel2note, 
                             f0=f0, ref_vari=tgt_vari,
                             infer=infer)
        if infer:
            return output
        losses = {}
        vari_gt = tgt_vari.unsqueeze(-1).repeat(*self.vari_repeat)
        add_mel_loss(output, vari_gt, losses, loss_and_lambda=self.loss_and_lambda)
        if not return_output:
            return losses
        else:
            return losses, output
    
    def validation_step(self, sample, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], _ = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < self.hparams['num_valid_plots']:
            vari_pred = self.run_model(sample, return_output=True, infer=True)
            vari_gt = sample[self.vari_type]
            self.plot_vari_spec(batch_idx, vari_gt, vari_pred)
        return outputs
    
    def plot_vari_spec(self, batch_idx, vari_tgt, vari_pred):
        name = f'vari_{batch_idx}'
        self.logger.add_figure(name, spec_curse_to_figure(vari_tgt[0], vari_pred[0]), self.global_step)

class VoicingPredictorTask(VariPredictorTask):
    def __init__(self, hparams):
        super().__init__(hparams, "voicing")

    def get_dataset_cls(self):
        return VoicingPredictorDataset
    
    def build_model(self):
        self.model = VoicingPredictor(self.hparams)

class BreathPredictorTask(VariPredictorTask):
    def __init__(self, hparams):
        super().__init__(hparams, "breath")

    def get_dataset_cls(self):
        return BreathPredictorDataset
    
    def build_model(self):
        self.model = BreathPredictor(self.hparams)