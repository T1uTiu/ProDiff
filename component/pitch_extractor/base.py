from typing import Dict


class BasePitchExtractor:
    def __init__(self, hparams):
        self.hparams = hparams
    
    def get_pitch(self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False):
        raise NotImplementedError
    
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