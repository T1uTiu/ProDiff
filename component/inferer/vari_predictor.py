import os
from component.inferer.base import Inferer, register_inferer
from modules.variance_predictor.vari_predictor import VoicingPredictor, BreathPredictor
from utils.ckpt_utils import load_ckpt


class VariPredictorInferer(Inferer):
    def build_model(self):
        model = PitchPredictor(self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model')
        model.to(self.device)
        self.model = model
    
    def run_model(self, **inp):
        note_midi = inp['note_midi']
        note_rest = inp['note_rest']
        mel2note = inp['mel2note']
        f0 = inp['f0']
        return self.model(note_midi, note_rest, mel2note, f0, infer=True)
    
    @staticmethod
    def category():
        return "pitch"

@register_inferer
class VoicingPredictorInferer(VariPredictorInferer):
    def build_model(self):
        model = VoicingPredictor(self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model')
        model.to(self.device)
        self.model = model

    @staticmethod
    def category():
        return "voicing"

@register_inferer
class BreathPredictorInferer(VariPredictorInferer):
    def build_model(self):
        model = BreathPredictor(self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], 'model')
        model.to(self.device)
        self.model = model

    @staticmethod
    def category():
        return "breath"