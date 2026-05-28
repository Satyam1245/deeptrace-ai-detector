import librosa
import numpy as np
import soundfile as sf

def load_audio(audio_path, target_sr=16000):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception:
        y, sr = sf.read(audio_path)
        if len(np.shape(y)) > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    if y.size == 0:
        raise ValueError("Audio file is empty")

    y = np.asarray(y, dtype=np.float32)
    return y, sr


def extract_mfcc(audio_path):
    y, sr = load_audio(audio_path, target_sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.normalize(mfcc)
    return mfcc


def extract_audio_heuristics(audio_path):
    y, sr = load_audio(audio_path, target_sr=16000)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
        voiced = f0[np.isfinite(f0)]
    except Exception:
        voiced = np.array([], dtype=np.float32)

    harmonic, percussive = librosa.effects.hpss(y)

    pitch_std = float(np.std(voiced)) if voiced.size else 0.0
    rms_std = float(np.std(rms))
    zcr_std = float(np.std(zcr))
    mfcc_delta_std = float(np.std(mfcc_delta))
    contrast_std = float(np.std(spectral_contrast))
    voiced_fraction = float(voiced.size / max(len(f0), 1)) if 'f0' in locals() else 0.0
    harmonic_rms = float(np.mean(librosa.feature.rms(y=harmonic)))
    percussive_rms = float(np.mean(librosa.feature.rms(y=percussive)))
    music_energy = harmonic_rms + percussive_rms

    suspicious_pitch = float(np.clip((35.0 - pitch_std) / 35.0, 0.0, 1.0))
    suspicious_rms = float(np.clip((0.03 - rms_std) / 0.03, 0.0, 1.0))
    suspicious_zcr = float(np.clip((0.025 - zcr_std) / 0.025, 0.0, 1.0))
    suspicious_mfcc = float(np.clip((6.0 - mfcc_delta_std) / 6.0, 0.0, 1.0))
    suspicious_contrast = float(np.clip((10.0 - contrast_std) / 10.0, 0.0, 1.0))
    speech_like_fake_prob = float(np.clip((0.12 - music_energy) / 0.12, 0.0, 1.0))

    heuristic_fake_prob = float(np.mean([
        suspicious_pitch,
        suspicious_rms,
        suspicious_zcr,
        suspicious_mfcc,
    ]))

    final_fake_prob = float(max(
        speech_like_fake_prob,
        heuristic_fake_prob * 0.5,
        suspicious_contrast * 0.25,
    ))

    if music_energy < 0.10:
        final_fake_prob = max(final_fake_prob, 0.65)

    if music_energy > 0.16:
        final_fake_prob = min(final_fake_prob, 0.15)

    return {
        "heuristic_fake_prob": heuristic_fake_prob,
        "final_fake_prob": final_fake_prob,
        "speech_like_fake_prob": speech_like_fake_prob,
        "pitch_std": pitch_std,
        "rms_std": rms_std,
        "zcr_std": zcr_std,
        "mfcc_delta_std": mfcc_delta_std,
        "contrast_std": contrast_std,
        "voiced_fraction": voiced_fraction,
        "harmonic_rms": harmonic_rms,
        "percussive_rms": percussive_rms,
        "music_energy": music_energy,
    }
