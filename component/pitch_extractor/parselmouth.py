import numpy as np
import parselmouth

from component.pitch_extractor.base import register_pe, BasePitchExtractor
from utils.data_gen_utils import pad_frames
from utils.pitch_utils import interp_f0

@register_pe
class Parselmouth(BasePitchExtractor):
    def __init__(self, hparams):
        self.hparams = hparams

    def get_pitch(self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False):
        time_step = hop_size / samplerate

        f0 = parselmouth.Sound(waveform, samplerate).to_pitch_ac(
            time_step=time_step, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max
        ).selected_array['frequency'].astype(np.float32)
        f0 = pad_frames(f0, hop_size, waveform.shape[0], length)
        uv = f0 == 0
        if interp_uv:
            f0, uv = interp_f0(f0, uv, self.hparams)
        return f0, uv