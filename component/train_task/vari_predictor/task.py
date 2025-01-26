import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_sepc_loss_prodiff
from modules.variance_predictor.vari_predictor import VariPredictor
from .dataset import VariPredictorDataset
import utils
from utils.plot import spec_curse_to_figure


class VariPredictorTask(BaseTask):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.build_phone_encoder()
        self.vari_prediction_args = hparams['vari_prediction_args']
        self.pred_voicing = hparams.get("use_voicing_embed", False)
        self.pred_breath = hparams.get("use_breath_embed", False)
        self.pred_tension = hparams.get("use_tension_embed", False)
        self.variance_list = []
        if self.pred_voicing:
            self.variance_list.append("voicing")
        if self.pred_breath:
            self.variance_list.append("breath")
        if self.pred_tension:
            self.variance_list.append("tension")
        
        repeat_bins = self.vari_prediction_args["repeat_bins"] // len(self.variance_list)
        if len(self.variance_list) == 1:
            self.vari_repeat = [1, 1, repeat_bins]
        else:
            self.vari_repeat = [1, 1, 1, repeat_bins]
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
        print("| losses:", self.loss_and_lambda)

    def get_dataset_cls(self):
        return VariPredictorDataset

    def build_model(self):
        self.model = VariPredictor(len(self.ph_encoder), self.hparams)
        utils.num_params(self.model)
        return self.model

    def run_model(self, sample, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_ph]
        mel2ph = sample["mel2ph"]
        note_midi = sample["note_midi"] 
        note_rest = sample["note_rest"]
        mel2note = sample["mel2note"]
        f0 = sample["f0"]
        spk_id = sample.get("spk_id", None)
        voicing = sample.get("voicing", None)
        breath = sample.get("breath", None)
        tension = sample.get("tension", None)
        # 模型输出
        output = self.model(
            txt_tokens, mel2ph,
            note_midi, note_rest, mel2note, 
            f0,
            spk_embed_id=spk_id,
            infer=infer,
            voicing=voicing, breath=breath, tension=tension
        )
        if infer:
            return output
        losses = {}
        tgt_vari = self.get_tgt_vari_spec(sample)
        add_sepc_loss_prodiff(output, tgt_vari, losses, loss_type=self.loss_and_lambda)
        if not return_output:
            return losses
        else:
            return losses, output
    
    def get_tgt_vari_spec(self, sample):
        tgt_vari = [sample[name] for name in self.variance_list]
        tgt_vari = torch.stack(tgt_vari, dim=1)
        if len(self.variance_list) == 1:
            tgt_vari = tgt_vari.squeeze(1)
        vari_gt = tgt_vari.unsqueeze(-1).repeat(*self.vari_repeat)
        return vari_gt

    def validation_step(self, sample, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], _ = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < self.hparams['num_valid_plots']:
            vari_pred = self.run_model(sample, return_output=True, infer=True)
            for name, pred_value in vari_pred.items():
                vari_gt = sample[name]
                self.plot_vari_spec(batch_idx, vari_gt, pred_value)
        return outputs
    
    def plot_vari_spec(self, batch_idx, vari_tgt, vari_pred, name):
        name = f'{name}_{batch_idx}'
        self.logger.add_figure(name, spec_curse_to_figure(vari_tgt[0], vari_pred[0]), self.global_step)