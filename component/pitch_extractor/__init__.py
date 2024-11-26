from typing import Dict
from component.pitch_extractor.base import BasePitchExtractor

PITCHEXTRACTORS: Dict[str, BasePitchExtractor] = {}
def register_pe(cls):
    PITCHEXTRACTORS[cls.__name__.lower()] = cls
    PITCHEXTRACTORS[cls.__name__] = cls
    return cls

def get_pitch_extractor(hparams):
    cls_name = hparams['pitch_extractor'].lower()
    if cls_name not in PITCHEXTRACTORS:
        raise ValueError(f"Unknown pitch extractor: {hparams['pitch_extractor']}")
    return PITCHEXTRACTORS[hparams['pitch_extractor'].lower()](hparams)
