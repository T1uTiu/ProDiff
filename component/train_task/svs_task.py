from component.train_task.dataset import SVSDataset
from modules.ProDiff.prodiff_teacher import ProDiffTeacher
import utils
from utils.hparams import hparams
from tasks.tts.fs2 import FastSpeech2Task
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder



class SVSTask(FastSpeech2Task):
    def __init__(self):
        super(SVSTask, self).__init__()
        self.dataset_cls = SVSDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def build_model(self):
        self.build_tts_model()
        utils.num_params(self.model) # 打印模型参数量
        return self.model

    def build_tts_model(self):
        self.model = ProDiffTeacher(self.phone_encoder, hparams)


    def run_model(self, model, sample: dict, return_output=False, infer=False):
        txt_tokens = sample["ph_seq"]  # [B, T_t]
        target = sample["mel"]  # [B, T_s, 80]
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        spk_embed_id = sample.get("spk_id", None)
        gender_embed_id = sample.get("gender_id", None)
        lang_seq = sample.get("lang_seq", None)
        # 模型输出
        output = model(txt_tokens, mel2ph, f0, 
                       lang_seq=lang_seq, spk_embed_id=spk_embed_id, gender_embed_id=gender_embed_id,
                       ref_mels=target, infer=infer)

        losses = {}
        self.add_mel_loss(output, target, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample: dict, batch_idx):
        outputs = {}
        txt_tokens = sample["ph_seq"]  # [B, T_t]

        spk_embed_id = sample.get("spk_id", None)
        gender_embed_id = sample.get("gender_id", None)
        lang_seq = sample.get("lang_seq", None)
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, mel2ph, f0, 
                lang_seq=lang_seq, spk_embed_id=spk_embed_id, gender_embed_id=gender_embed_id,
                ref_mels=None, infer=True)
            self.plot_mel(batch_idx, sample.mel, model_out)
        return outputs

