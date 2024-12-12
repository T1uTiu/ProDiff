import os

import torch
from component.inferer.base import Inferer, register_inferer
from modules.variance_predictor.dur_predictor import DurPredictor
from utils.ckpt_utils import load_ckpt

@register_inferer
class DurPredictorInferer(Inferer):
    def build_model(self, ph_encoder):
        model = DurPredictor(ph_encoder, self.hparams)
        model.eval()
        work_dir = os.path.join("checkpoints", self.hparams['exp_name'], "dur")
        load_ckpt(model, work_dir, 'model')
        model.to(self.device)
        self.model = model
    
    def run_model(self, **inp):
        ph_seq = inp['ph_seq']
        onset = inp['onset']
        word_dur = inp['word_dur']
        dur_pred = self.model(ph_seq, onset, word_dur)
        ph_num = inp['ph_num']
        note_dur = inp['note_dur']
        return self.force_align_pdur(ph_num, dur_pred.squeeze(0), note_dur)
    
    def force_align_pdur(self, ph_num, ph_dur, note_dur):
        """
        ph_num: torch.LongTensor, [T_note]
        ph_dur: torch.FloatTensor, [T_ph]
        note_dur: list, [T_note]
        """
        note_num = len(note_dur)
        j = 0
        for i in range(note_num):
            pn = ph_num[i].item()
            rate = torch.sum(ph_dur[j:j+pn]) / note_dur[i]
            ph_dur[j:j+ph_num[i]] = ph_dur[j:j+ph_num[i]] / rate
            j += pn
        return ph_dur
    
    @staticmethod
    def category():
        return "dur"