class BaseVocoder:
    def __init__(self, hparams):
        self.hparams = hparams
        
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn, hparams):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError

VOCODERS = {}
def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(vocoder):
    cls_name = vocoder.lower()
    if cls_name not in VOCODERS:
        raise ValueError(f"Vocoder {cls_name} not found in VOCODERS")
    return VOCODERS[vocoder]



