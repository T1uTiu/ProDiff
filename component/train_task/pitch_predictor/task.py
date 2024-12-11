from component.train_task.base_task import BaseTask
from component.train_task.pitch_predictor.dataset import PitchPredictorDataset
from modules.variance_predictor.pitch_predictor import PitchPredictor


class PitchPredictorTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.build_phone_encoder()  

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
        dur_pred = self.model(txt_tokens, mel2ph, base_f0, f0, spk_id=spk_id)

        losses = {}
        if not return_output:
            return losses
        else:
            return losses, dur_pred
    
    def validation_step(self, sample, batch_idx):
        return super().validation_step(sample, batch_idx)