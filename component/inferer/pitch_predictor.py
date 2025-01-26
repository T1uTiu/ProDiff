import os
from component.inferer.base import Inferer, register_inferer
from modules.variance_predictor.pitch_predictor import PitchPredictor
from utils.ckpt_utils import load_ckpt

@register_inferer
class PitchPredictorInferer(Inferer):
    def build_model(self, ph_category_encoder):
        model = PitchPredictor(len(ph_category_encoder), self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model')
        model.to(self.device)
        self.model = model
    
    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        mel2ph = inp['mel2ph']
        note_midi = inp['note_midi']
        note_rest = inp['note_rest']
        mel2note = inp['mel2note']
        base_pitch = inp['base_pitch']
        pitch_expr = inp["pitch_expr"]
        spk_id = inp.get('spk_id', None)
        return self.model(ph_seq, mel2ph, note_midi, note_rest, mel2note, base_pitch, pitch_expr=pitch_expr, spk_id=spk_id, infer=True)
    
    @staticmethod
    def category():
        return "pitch"