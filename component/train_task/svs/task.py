import torch
from component.train_task.base_task import BaseTask
from component.train_task.loss_utils import add_sepc_loss_prodiff, add_spec_loss_reflow
from component.train_task.svs.dataset import SVSDataset, SVSRectifiedDataset
from modules.decoder.wavenet import WaveNet
from modules.diffusion.prodiff import GaussianDiffusion
from modules.diffusion.reflow import RectifiedFlow
from modules.svs.prodiff_teacher import ProDiffTeacher
import utils
from utils.ckpt_utils import load_ckpt
from utils.plot import spec_to_figure

class SVSTask(BaseTask):
    def __init__(self, hparams):
        super(SVSTask, self).__init__(hparams=hparams)
        self.diffusion_type = hparams.get("diff_type", "prodiff")
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_type_list = mel_losses
        self.loss_type = {}
        for l in mel_losses:
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_type[l] = lbd
        print("| Mel losses:", self.loss_type)
        
        self.build_phone_encoder()

    def get_dataset_cls(self):
        return SVSDataset

    def build_model(self):
        self.model = ProDiffTeacher(len(self.ph_encoder), self.hparams)
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def run_model(self, sample: dict, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_t]
        gt_mel = sample["mel"]  # [B, T_s, 80]
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        spk_embed_id = sample.get("spk_id", None)
        gender_embed_id = sample.get("gender_id", None)
        lang_seq = sample.get("lang_seq", None)
        voicing = sample.get("voicing", None)
        breath = sample.get("breath", None)
        # 模型输出
        output = self.model(
            txt_tokens, mel2ph, f0, 
            lang_seq=lang_seq, 
            spk_embed_id=spk_embed_id, 
            gender_embed_id=gender_embed_id,
            voicing=voicing, breath=breath,
            gt_spec=gt_mel, infer=infer
        )
        if infer:
            return output
        losses = {}
        spec_pred, spec_gt, t = output
        non_padding = (mel2ph > 0).unsqueeze(-1)
        if self.diffusion_type == "prodiff":
            add_sepc_loss_prodiff(
                spec_pred, spec_gt, non_padding, 
                loss_type=self.loss_type,
                losses=losses, name="mel"
            )
        elif self.diffusion_type == "reflow":
            add_spec_loss_reflow(
                spec_pred, spec_gt, t, non_padding, 
                self.loss_type_list[0], log_norm=True, 
                losses=losses, name="mel"
            )
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
        if batch_idx < self.hparams['num_valid_plots']:
            model_out = self.run_model(sample, return_output=True, infer=True)
            self.plot_mel(batch_idx, sample["mel"], model_out)
        return outputs
    
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else f"{name}_{batch_idx}"
        vmin = self.hparams['mel_vmin']
        vmax = self.hparams['mel_vmax']
        self.logger.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

class SVSSRecitifedTask(SVSTask):
    def get_dataset_cls(self):
        return SVSRectifiedDataset

    def build_model(self):
        hparams = self.hparams
        # train model
        self.diffusion_type = hparams.get("diff_type", "prodiff")
        if self.diffusion_type == "prodiff":
            self.model = GaussianDiffusion(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
                    in_dims=hparams['audio_num_mel_bins'],
                    hidden_size=hparams["hidden_size"],
                    residual_layers=hparams["residual_layers"],
                    residual_channels=hparams["residual_channels"],
                    dilation_cycle_length=hparams["dilation_cycle_length"],
                ),
                timesteps=hparams["timesteps"],
                time_scale=hparams["timescale"],
                schedule_type=hparams['schedule_type'],
                max_beta=hparams.get("max_beta", 0.06),
                spec_min=hparams["spec_min"],
                spec_max=hparams["spec_max"],
            )
        elif self.diffusion_type == "reflow":
            self.model = RectifiedFlow(
                out_dims=hparams["audio_num_mel_bins"],
                denoise_fn=WaveNet(
                    in_dims=hparams['audio_num_mel_bins'],
                    hidden_size=hparams["hidden_size"],
                    residual_layers=hparams["residual_layers"],
                    residual_channels=hparams["residual_channels"],
                    dilation_cycle_length=hparams["dilation_cycle_length"],
                ),
                time_scale=hparams["timescale"],
                num_features=1,
                sampling_algorithm=hparams.get("sampling_algorithm", "euler"),
                spec_min=hparams["spec_min"],
                spec_max=hparams["spec_max"],
            )
        utils.num_params(self.model)
        return self.model
    
    def run_model(self, sample, return_output=False, infer=False):
        mel2ph = sample["mel2ph"]
        condition = sample["condition"] # [B, T, hidden]
        x_T, x_0 = sample["x_T"], sample["x_0"] # [B, 1, M, T]
        output = self.model.forward(condition, x_T, gt_spec=x_0, infer=infer)
        if infer:
            return output
        losses = {}
        spec_pred, spec_gt, t = output
        non_padding = (mel2ph > 0).unsqueeze(-1)
        if self.diffusion_type == "prodiff":
            add_sepc_loss_prodiff(
                spec_pred, spec_gt, non_padding, 
                loss_type=self.loss_type,
                losses=losses, name="mel"
            )
        elif self.diffusion_type == "reflow":
            add_spec_loss_reflow(
                spec_pred, spec_gt, t, non_padding, 
                self.loss_type_list[0], log_norm=True, 
                losses=losses, name="mel"
            )
        if not return_output:
            return losses
        else:
            return losses, output