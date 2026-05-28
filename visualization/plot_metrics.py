"""RRMSE / CC metrics grouped by SNR for model comparison."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def _rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def _rrmse_temporal(pred, clean):
    return _rms(pred - clean) / _rms(clean)


def _rrmse_spectral(pred, clean, fs):
    vals = []
    for p, c in zip(pred, clean):
        _, psd_p = welch(p, fs=fs, nperseg=min(256, len(p)))
        _, psd_c = welch(c, fs=fs, nperseg=min(256, len(c)))
        vals.append(_rms(psd_p - psd_c) / _rms(psd_c))
    return np.array(vals)


def _correlation_coeff(pred, clean):
    vals = []
    for p, c in zip(pred, clean):
        if np.std(p) == 0 or np.std(c) == 0:
            vals.append(np.nan)
        else:
            vals.append(np.corrcoef(p, c)[0, 1])
    return np.array(vals)


def _group_by_snr(values, snr_num=10):
    return np.array_split(values, snr_num)


def compute_metrics(pred, clean, fs):
    """Compute RRMSE (temporal & spectral) and correlation coefficient per SNR group."""
    rt = _rrmse_temporal(pred, clean)
    rs = _rrmse_spectral(pred, clean, fs)
    cc = _correlation_coeff(pred, clean)

    return {
        "rrmse_temporal": np.array([np.nanmean(x) for x in _group_by_snr(rt)]),
        "rrmse_spectral": np.array([np.nanmean(x) for x in _group_by_snr(rs)]),
        "cc": np.array([np.nanmean(x) for x in _group_by_snr(cc)]),
    }


def plot_metrics_by_snr(result_dir, noise_type, label, save_path):
    """Plot three metric panels vs SNR for one denoising method."""
    fs = 256 if noise_type == "EOG" else 512
    snrs = np.arange(-7, 3)

    clean = np.load(os.path.join(result_dir, "EEG_test.npy"))
    denoised = np.load(os.path.join(result_dir, "Denoiseoutput_test.npy"))

    metrics = compute_metrics(denoised, clean, fs)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(snrs, metrics["rrmse_temporal"], marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("RRMSE temporal")
    plt.title("Temporal RRMSE")
    plt.legend([label])

    plt.subplot(1, 3, 2)
    plt.plot(snrs, metrics["rrmse_spectral"], marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("RRMSE spectral")
    plt.title("Spectral RRMSE")
    plt.legend([label])

    plt.subplot(1, 3, 3)
    plt.plot(snrs, metrics["cc"], marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("CC")
    plt.title("Correlation coefficient")
    plt.legend([label])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    np.save(os.path.join(result_dir, "metrics_by_snr.npy"), metrics)
    print("Saved:", save_path)


def plot_multi_metrics_by_snr(result_dir, noise_type, output_dict, save_path):
    """Plot three metric panels vs SNR for multiple methods on the same figure.

    Parameters
    ----------
    output_dict : dict[str, np.ndarray]
        Mapping of method name to denoised output array.
    """
    fs = 256 if noise_type == "EOG" else 512
    snrs = np.arange(-7, 3)
    clean = np.load(os.path.join(result_dir, "EEG_test.npy"))

    all_metrics = {}
    for name, pred in output_dict.items():
        all_metrics[name] = compute_metrics(pred, clean, fs)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    for name, m in all_metrics.items():
        plt.plot(snrs, m["rrmse_temporal"], marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RRMSE temporal")
    plt.title("Temporal RRMSE")
    plt.legend()

    plt.subplot(1, 3, 2)
    for name, m in all_metrics.items():
        plt.plot(snrs, m["rrmse_spectral"], marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RRMSE spectral")
    plt.title("Spectral RRMSE")
    plt.legend()

    plt.subplot(1, 3, 3)
    for name, m in all_metrics.items():
        plt.plot(snrs, m["cc"], marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("CC")
    plt.title("Correlation coefficient")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    np.save(os.path.join(result_dir, "metrics_by_snr.npy"), all_metrics)
    print("Saved:", save_path)
