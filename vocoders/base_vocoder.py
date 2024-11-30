import importlib
from typing import Dict

class BaseVocoder:
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError

VOCODERS: Dict[str, BaseVocoder] = {}
def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(hparams):
    cls_name = hparams['vocoder'].lower()
    if cls_name not in VOCODERS:
        raise ValueError(f"Vocoder {cls_name} not found in VOCODERS")
    return VOCODERS[hparams['vocoder']]



