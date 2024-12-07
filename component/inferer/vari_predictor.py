import os
from component.inferer.base import Inferer, register_inferer
from modules.variance_predictor.vari_predictor import VariPredictor
from utils.ckpt_utils import load_ckpt

@register_inferer
class VariPredictorInferer(Inferer):
    def build_model(self, ph_encoder):
        model = VariPredictor(ph_encoder, self.hparams)
        model.eval()
        work_dir = os.path.join("checkpoints", self.hparams['exp_name'], "vari")
        load_ckpt(model, work_dir, 'model')
        model.to(self.device)
        self.model = model
    
    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        onset = inp['onset']
        word_dur = inp['word_dur']
        return self.model.predict_dur(ph_seq, onset, word_dur)
    
    @staticmethod
    def category():
        return "vari"