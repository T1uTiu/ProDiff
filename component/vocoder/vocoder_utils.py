import librosa

import numpy as np


def denoise(wav, v=0.1, fft_size=2048, hop_size=512, win_size=512):
    spec = librosa.stft(y=wav, n_fft=fft_size, hop_length=hop_size,
                        win_length=win_size, pad_mode='constant')
    spec_m = np.abs(spec)
    spec_m = np.clip(spec_m - v, a_min=0, a_max=None)
    spec_a = np.angle(spec)

    return librosa.istft(spec_m * np.exp(1j * spec_a), hop_length=hop_size,
                         win_length=win_size)
