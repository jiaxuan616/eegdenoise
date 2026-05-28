"""Butterworth-based filter baselines for EEG denoising."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _butter_filter_1d(x, fs, lowcut=None, highcut=None, order=4):
    nyq = 0.5 * fs
    if lowcut is not None and highcut is not None:
        btype = "bandpass"
        wn = [lowcut / nyq, highcut / nyq]
    elif lowcut is not None:
        btype = "highpass"
        wn = lowcut / nyq
    elif highcut is not None:
        btype = "lowpass"
        wn = highcut / nyq
    else:
        raise ValueError("lowcut and highcut cannot both be None.")
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)


def _notch_filter_1d(x, fs, freq=50.0, q=30.0):
    b, a = iirnotch(w0=freq / (fs / 2), Q=q)
    return filtfilt(b, a, x)


def _apply_to_batch(data, func):
    data = np.asarray(data)
    if data.ndim == 1:
        return func(data)
    return np.array([func(x) for x in data])


# ---- Pre-processing helpers (used before mixing) ----

def preprocess_clean_eeg_segments(eeg, fs, powerline=50.0):
    """Preprocess clean EEG: 1-80 Hz band-pass + notch."""
    def func(x):
        y = _butter_filter_1d(x, fs=fs, lowcut=1, highcut=80)
        y = _notch_filter_1d(y, fs=fs, freq=powerline)
        return y
    return _apply_to_batch(eeg, func)


def preprocess_eog_segments(eog, fs):
    """Preprocess EOG: 0.3-10 Hz band-pass."""
    return _apply_to_batch(eog, lambda x: _butter_filter_1d(x, fs=fs, lowcut=0.3, highcut=10))


def preprocess_emg_segments(emg, fs, powerline=50.0):
    """Preprocess EMG: 1-120 Hz band-pass + notch."""
    def func(x):
        y = _butter_filter_1d(x, fs=fs, lowcut=1, highcut=120)
        y = _notch_filter_1d(y, fs=fs, freq=powerline)
        return y
    return _apply_to_batch(emg, func)


# ---- Denoising baselines ----

def filter_denoise(noisy_eeg, noise_type, fs):
    """Traditional filter baseline for already mixed noisy EEG.

    Paper baselines:
      - EOG noisy EEG  → high-pass 12 Hz
      - EMG noisy EEG  → band-pass 12-40 Hz
      - EOG_EMG noisy EEG → high-pass 12 Hz (simple extension)
    """
    noise_type = noise_type.upper()
    noisy_eeg = np.asarray(noisy_eeg)

    if noise_type == "EOG":
        return _apply_to_batch(noisy_eeg, lambda x: _butter_filter_1d(x, fs=fs, lowcut=12))
    elif noise_type == "EMG":
        return _apply_to_batch(noisy_eeg, lambda x: _butter_filter_1d(x, fs=fs, lowcut=12, highcut=40))
    elif noise_type == "EOG_EMG":
        return _apply_to_batch(noisy_eeg, lambda x: _butter_filter_1d(x, fs=fs, lowcut=12))
    else:
        raise ValueError("noise_type must be 'EOG', 'EMG', or 'EOG_EMG'.")
