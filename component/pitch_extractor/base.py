from abc import ABC, abstractmethod


class PitchExtractor(ABC):
    @abstractmethod
    def get_pitch(self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False):
        pass