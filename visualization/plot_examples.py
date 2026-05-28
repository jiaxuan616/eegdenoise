"""Plot example waveforms and PSDs for noisy / clean / denoised signals."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def load_result(result_dir, denoised_file=None):
    """Load noisy, denoised, and clean EEG arrays from result_dir."""
    noisy = np.load(os.path.join(result_dir, "noiseinput_test.npy"))
    clean = np.load(os.path.join(result_dir, "EEG_test.npy"))
    if denoised_file is None:
        denoised_file = "Denoiseoutput_test.npy"
    denoised = np.load(os.path.join(result_dir, denoised_file))
    return noisy, denoised, clean


def plot_example_waveform_psd(noisy, denoised, clean, fs, index, save_path):
    """Plot time-domain waveforms and PSD for a single example."""
    t = np.arange(clean.shape[1]) / fs

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, clean[index], label="Ground-truth EEG")
    plt.plot(t, noisy[index], label="Noisy EEG")
    plt.plot(t, denoised[index], label="Denoised EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Temporal domain, index={index}")
    plt.legend()

    f_clean, p_clean = welch(clean[index], fs=fs, nperseg=min(256, clean.shape[1]))
    f_noisy, p_noisy = welch(noisy[index], fs=fs, nperseg=min(256, noisy.shape[1]))
    f_denoised, p_denoised = welch(denoised[index], fs=fs, nperseg=min(256, denoised.shape[1]))

    plt.subplot(2, 1, 2)
    plt.semilogy(f_clean, p_clean, label="Ground-truth EEG")
    plt.semilogy(f_noisy, p_noisy, label="Noisy EEG")
    plt.semilogy(f_denoised, p_denoised, label="Denoised EEG")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Frequency domain")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
