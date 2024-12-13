import os
from component.inferer.base import Inferer, register_inferer
from modules.variance_predictor.pitch_predictor import PitchPredictor
from utils.ckpt_utils import load_ckpt

@register_inferer
class PitchPredictorInferer(Inferer):
    def build_model(self):
        model = PitchPredictor(self.hparams)
        model.eval()
        work_dir = os.path.join("checkpoints", self.hparams['exp_name'], "pitch")
        load_ckpt(model, work_dir, 'model')
        model.to(self.device)
        self.model = model
    
    def run_model(self, **inp):
        note_midi = inp['note_midi']
        mel2note = inp['mel2note']
        base_f0 = inp['base_f0']
        spk_id = inp.get('spk_id', None)
        return self.model(note_midi, mel2note, base_f0, spk_id=spk_id, infer=True)
    
    @staticmethod
    def category():
        return "pitch"