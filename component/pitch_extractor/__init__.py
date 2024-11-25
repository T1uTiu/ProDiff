from typing import Dict
from component.pitch_extractor.base import BasePitchExtractor
from component.pitch_extractor.rmvpe import RMVPE
from component.pitch_extractor.parselmouth import Parselmouth

PITCHEXTRACTORS: Dict[str, BasePitchExtractor] = {}
def register_pe(cls):
    PITCHEXTRACTORS[cls.__name__.lower()] = cls
    PITCHEXTRACTORS[cls.__name__] = cls
    return cls

def init_pitch_extractor(hparams):
    if hparams['pitch_extractor'] == 'rmvpe':
        return RMVPE(hparams['pe_ckpt'],  hparams=hparams)
    elif hparams['pitch_extractor'] == 'parselmouth':
        return Parselmouth(hparams)
    else:
        raise ValueError(f"Unknown pitch extractor: {hparams['pitch_extractor']}")