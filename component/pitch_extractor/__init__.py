from component.pitch_extractor.base import PitchExtractor
from component.pitch_extractor.rmvpe import RMVPE
from component.pitch_extractor.parselmouth import Parselmouth


def init_pitch_extractor(hparams):
    if hparams['pitch_extractor'] == 'rmvpe':
        return RMVPE(hparams['pe_ckpt'], is_half=False, hparams=hparams)
    elif hparams['pitch_extractor'] == 'parselmouth':
        return Parselmouth(hparams)
    else:
        raise ValueError(f"Unknown pitch extractor: {hparams['pitch_extractor']}")