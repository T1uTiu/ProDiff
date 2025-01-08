import csv
import json
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from modules.nsf_hifigan.nvSTFT import STFT
from utils.pitch_utils import interp_f0
from utils.text_encoder import TokenTextEncoder
from modules.vr import load_sep_model


def build_phone_encoder(data_dir: str, dictionary: dict):
    ph2global = {}
    if dictionary.get("global", None):
        f = open(dictionary["global"], 'r')
        for label in csv.DictReader(f):
            for lang, ph in label.items():
                if lang == "global":
                    continue
                ph2global[f"{ph}/{lang}"] = label["global"]
        f.close()

    ph_set_fn = f"{data_dir}/phone_set.json"
    ph_map = {}
    if not os.path.exists(ph_set_fn):
        for lang, dictionary in dictionary.items():
            if lang == "global":
                continue
            f = open(dictionary, 'r')
            ph_map[f"AP/{lang}"] = "AP"
            ph_map[f"SP/{lang}"] = "SP"
            for x in f.readlines():
                ph_list = x.split("\n")[0].split('\t')[1].split(' ')
                for ph in ph_list:
                    ph = f"{ph}/{lang}"
                    ph_map[ph] = ph2global.get(ph, ph)
            f.close()
        json.dump(ph_map, open(ph_set_fn, 'w'))
    else:
        ph_map = json.load(open(ph_set_fn, 'r'))
    ph_list = list(sorted(set(ph_map.values())))
    print("| phone set: ", ph_list)
    ph_encoder = TokenTextEncoder(None, vocab_list=ph_list, replace_oov="SP")
    return ph_map, ph_encoder

def build_lang_map(data_dir, dictionary: dict):
    lang_map = {dt: i for i, dt in enumerate(dictionary.keys()) if dt != "global"}
    print("| lang_map: ", lang_map)
    lang_map_fn = f"{data_dir}/lang_map.json"
    with open(lang_map_fn, 'w') as f:
        json.dump(lang_map, f)
    return lang_map\
    
def build_spk_map(data_dir, datasets):
    spk_map = {ds["speaker"]: i for i, ds in enumerate(datasets)}
    print("| spk_map: ", spk_map)
    spk_map_fn = f"{data_dir}/spk_map.json"
    with open(spk_map_fn, 'w') as f:
        json.dump(spk_map, f)
    return spk_map

def get_mel_spec(waveform: np.ndarray, 
                 samplerate, num_mels, fft_size, win_size, hop_size, fmin, fmax, 
                   keyshift=0, speed=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stft = STFT(samplerate, num_mels, fft_size, win_size, hop_size, fmin, fmax)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device), keyshift=keyshift, speed=speed).squeeze(0).T
        # log mel to log10 mel
        mel_torch = 0.434294 * mel_torch
        return mel_torch.cpu().numpy()

VR_MODEL = None

def extract_harmonic_aperiodic(waveform, model_path):
    global VR_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if VR_MODEL is None:
        VR_MODEL = load_sep_model(model_path, device=device)
    # infer
    with torch.no_grad():
        x = torch.from_numpy(waveform).to(device).reshape(1, 1, -1)
        if not VR_MODEL.is_mono:
            x = x.repeat(1, 2, 1)
        x = VR_MODEL.predict_from_audio(x)
        x = torch.mean(x, dim=1)
        harmonic_part = x.squeeze().cpu().numpy()
        aperiodic_part = waveform - harmonic_part
    return harmonic_part, aperiodic_part

def get_energy(waveform, mel_len, hop_size, win_size, domain="db"):
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    if len(energy) < mel_len:
        energy = np.pad(energy, (0, mel_len - len(energy)))
    energy = energy[:mel_len]
    if domain == "db":
        energy = librosa.amplitude_to_db(energy)
    elif domain == "amplitude":
        pass
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return energy

def get_voicing(sp, mel_len, hop_size, win_size, smooth_func, norm=True, db_min=-96.0, db_max=-12.0, device="cuda"):
    voicing = get_energy(sp, mel_len, hop_size, win_size)
    voicing = smooth_func(torch.from_numpy(voicing).to(device)[None])[0]
    if norm:
        voicing = torch.clamp(voicing, db_min, db_max)
        voicing = (voicing - db_min) / (db_max - db_min)
    return voicing.detach().cpu().numpy()

def get_breath(ap, mel_len, hop_size, win_size, smooth_func, norm=True, db_min=-96.0, db_max=-12.0, device="cuda"):
    breath = get_energy(ap, mel_len, hop_size, win_size)
    breath = smooth_func(torch.from_numpy(breath).to(device)[None])[0]
    if norm:
        breath = torch.clamp(breath, db_min, db_max)
        breath = (breath - db_min) / (db_max - db_min)
    return breath.detach().cpu().numpy()

def get_kth_harmonic(k, harmonic_part, f0, hop_size, win_size, samplerate, half_width=3.5):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    waveform = torch.from_numpy(harmonic_part).unsqueeze(0).to(device)  # [B, n_samples]
    n_samples = waveform.shape[1]       
    f0 = f0 * (k + 1)
    pad_size = int(n_samples // hop_size) - len(f0) + 1
    if pad_size > 0:
        f0 = np.pad(f0, (0, pad_size), mode='constant', constant_values=(f0[0], f0[-1]))
    
    f0, _ = interp_f0(f0, uv=f0 == 0)
    f0 = torch.from_numpy(f0).to(device)[None, :, None]  # [B, n_frames, 1]
    n_f0_frames = f0.shape[1]

    phase = torch.arange(win_size, dtype=waveform.dtype, device=device) / win_size * 2 * np.pi
    nuttall_window = (
            0.355768
            - 0.487396 * torch.cos(phase)
            + 0.144232 * torch.cos(2 * phase)
            - 0.012604 * torch.cos(3 * phase)
    )
    spec = torch.stft(
        waveform,
        n_fft=win_size,
        win_length=win_size,
        hop_length=hop_size,
        window=nuttall_window,
        center=True,
        return_complex=True
    ).permute(0, 2, 1)  # [B, n_frames, n_spec]
    n_spec_frames, n_specs = spec.shape[1:]
    idx = torch.arange(n_specs).unsqueeze(0).unsqueeze(0).to(f0)  # [1, 1, n_spec]
    center = f0 * win_size / samplerate
    start = torch.clip(center - half_width, min=0)
    end = torch.clip(center + half_width, max=n_specs)
    idx_mask = (center >= 1) & (idx >= start) & (idx < end)  # [B, n_frames, n_spec]
    if n_f0_frames < n_spec_frames:
        idx_mask = F.pad(idx_mask, [0, 0, 0, n_spec_frames - n_f0_frames])
    spec = spec * idx_mask[:, :n_spec_frames, :]
    k_harmonic = torch.istft(
        spec.permute(0, 2, 1),
        n_fft=win_size,
        win_length=win_size,
        hop_length=hop_size,
        window=nuttall_window,
        center=True,
        length=n_samples
    ).squeeze(0).cpu().numpy()

    return k_harmonic

def get_tension(sp, mel_len, f0, hop_size, win_size, samplerate, smooth_func, half_width=3.5, domain="logit", device="cuda"):
    base_sp = get_kth_harmonic(0, sp, f0, hop_size, win_size, samplerate, half_width)
    # energy
    energy_full = get_energy(sp, mel_len, hop_size, win_size, domain="amplitude")
    energy_base = get_energy(base_sp, mel_len, hop_size, win_size, domain="amplitude")
    # tension
    tension = np.sqrt(np.clip(energy_full ** 2 - energy_base ** 2, a_min=0, a_max=None)) / (energy_full + 1e-5)
    if domain == 'ratio':
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == 'db':
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension)
    elif domain == 'logit':
        tension = np.clip(tension, a_min=1e-4, a_max=1 - 1e-4)
        tension = np.log(tension / (1 - tension))
    # smooth
    tension = smooth_func(torch.from_numpy(tension).to(device)[None])[0]
    return tension.detach().cpu().numpy()