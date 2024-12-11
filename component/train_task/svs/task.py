import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_mel_loss
from component.train_task.svs.dataset import SVSDataset
from modules.ProDiff.prodiff_teacher import ProDiffTeacher
import utils
from utils.hparams import hparams
from utils.plot import spec_to_figure

class SVSTask(BaseTask):
    def __init__(self, hparams):
        super(SVSTask, self).__init__(hparams=hparams)
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
        print("| Mel losses:", self.loss_and_lambda)
        self.build_phone_encoder()

    def get_dataset_cls(self):
        return SVSDataset

    def build_model(self):
        self.model = ProDiffTeacher(self.ph_encoder, hparams)
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def run_model(self, sample: dict, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_t]
        target = sample["mel"]  # [B, T_s, 80]
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        spk_embed_id = sample.get("spk_id", None)
        gender_embed_id = sample.get("gender_id", None)
        lang_seq = sample.get("lang_seq", None)
        # 模型输出
        output = self.model(txt_tokens, mel2ph, f0, 
                       lang_seq=lang_seq, spk_embed_id=spk_embed_id, gender_embed_id=gender_embed_id,
                       ref_mels=target, infer=infer)

        losses = {}
        add_mel_loss(output, target, losses, loss_and_lambda=self.loss_and_lambda)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample: dict, batch_idx):
        outputs = {
            "nsamples": sample["nsamples"],
        }
        outputs['losses'], model_out = self.run_model(sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            _, model_out = self.run_model(sample, return_output=True, infer=True)
            self.plot_mel(batch_idx, sample["mel"], model_out)
        return outputs
    
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.hparams['mel_vmin']
        vmax = self.hparams['mel_vmax']
        self.logger.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

