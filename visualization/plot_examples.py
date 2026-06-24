"""Example waveform and PSD plots for EEG denoising results.

This module keeps the original lightweight API and adds the useful example-plot
logic from the larger analysis scripts.  It intentionally does not contain
metric-report generation; those functions live in ``plot_metrics.py``.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Return arr as (n_samples, n_times)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def _load_array(path: str, *, allow_pickle: bool = True) -> np.ndarray:
    """Load a numpy array with a clear error message."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def _apply_std_if_needed(arr: np.ndarray, std_value: Optional[np.ndarray]) -> np.ndarray:
    """Reverse per-sample standardization when std_value is provided."""
    if std_value is None:
        return arr
    std_value = np.asarray(std_value)
    if std_value.ndim == 0:
        return arr * std_value
    return arr * std_value.reshape(-1, 1)


def load_result(
    result_dir: str,
    denoised_file: Optional[str] = None,
    *,
    noisy_file: str = "noiseinput_test.npy",
    clean_file: str = "EEG_test.npy",
    std_file: Optional[str] = None,
    apply_std: bool = False,
) -> ArrayTriplet:
    """Load noisy, denoised, and clean EEG arrays from a result directory.

    Parameters
    ----------
    result_dir:
        Directory containing EEGdenoiseNet output files.
    denoised_file:
        Denoised output filename. Defaults to ``Denoiseoutput_test.npy``.
        Use ``Filter_output_test.npy`` or ``EMD_output_test.npy`` for
        traditional baselines.
    noisy_file, clean_file:
        Input and target filenames.
    std_file:
        Optional standard-deviation filename. If omitted and ``apply_std=True``,
        ``test_std_VALUE.npy`` is used when it exists.
    apply_std:
        Whether to reverse per-sample standardization.

    Returns
    -------
    noisy, denoised, clean:
        Arrays in the same order as the original helper.
    """
    denoised_file = denoised_file or "Denoiseoutput_test.npy"

    noisy = _ensure_2d(_load_array(os.path.join(result_dir, noisy_file)))
    denoised = _ensure_2d(_load_array(os.path.join(result_dir, denoised_file)))
    clean = _ensure_2d(_load_array(os.path.join(result_dir, clean_file)))

    std_value = None
    if apply_std:
        std_file = std_file or "test_std_VALUE.npy"
        std_path = os.path.join(result_dir, std_file)
        if os.path.exists(std_path):
            std_value = _load_array(std_path)

    noisy = _apply_std_if_needed(noisy, std_value)
    denoised = _apply_std_if_needed(denoised, std_value)
    clean = _apply_std_if_needed(clean, std_value)
    return noisy, denoised, clean


def load_method_outputs(
    result_dir: str,
    *,
    method_files: Optional[Mapping[str, str]] = None,
    include_default: bool = True,
    apply_std: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load clean/noisy arrays plus one or more denoised method outputs.

    This replaces the repeated loading logic that appeared in several one-off
    comparison scripts.
    """
    method_files = dict(method_files or {})
    if include_default and os.path.exists(os.path.join(result_dir, "Denoiseoutput_test.npy")):
        method_files.setdefault("Denoised", "Denoiseoutput_test.npy")
    if os.path.exists(os.path.join(result_dir, "Filter_output_test.npy")):
        method_files.setdefault("Filter", "Filter_output_test.npy")
    if os.path.exists(os.path.join(result_dir, "EMD_output_test.npy")):
        method_files.setdefault("EMD", "EMD_output_test.npy")

    noisy, _, clean = load_result(
        result_dir,
        denoised_file=next(iter(method_files.values())) if method_files else "Denoiseoutput_test.npy",
        apply_std=apply_std,
    )
    outputs: dict[str, np.ndarray] = {}
    for name, filename in method_files.items():
        _, denoised, _ = load_result(result_dir, filename, apply_std=apply_std)
        outputs[name] = denoised
    return noisy, clean, outputs


def _rrmse_per_sample(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    pred = _ensure_2d(pred)
    clean = _ensure_2d(clean)
    err = np.sqrt(np.mean((pred - clean) ** 2, axis=1))
    ref = np.sqrt(np.mean(clean ** 2, axis=1)) + 1e-12
    return err / ref


def _plot_waveforms(ax, t: np.ndarray, clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> None:
    ax.plot(t, clean, label="Ground-truth EEG", linewidth=1.2)
    ax.plot(t, noisy, label="Noisy EEG", alpha=0.7, linewidth=1.0)
    ax.plot(t, denoised, label="Denoised EEG", alpha=0.9, linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="best")


def _plot_psd(ax, clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, fs: float) -> None:
    nperseg = min(256, len(clean))
    f_clean, p_clean = welch(clean, fs=fs, nperseg=nperseg)
    f_noisy, p_noisy = welch(noisy, fs=fs, nperseg=min(256, len(noisy)))
    f_denoised, p_denoised = welch(denoised, fs=fs, nperseg=min(256, len(denoised)))

    ax.semilogy(f_clean, p_clean, label="Ground-truth EEG", linewidth=1.2)
    ax.semilogy(f_noisy, p_noisy, label="Noisy EEG", alpha=0.7, linewidth=1.0)
    ax.semilogy(f_denoised, p_denoised, label="Denoised EEG", alpha=0.9, linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(loc="best")


def _minus_6db_frequency(x: np.ndarray, fs: float) -> float:
    """Return the first post-peak frequency where PSD drops below -6 dB."""
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    peak_idx = int(np.argmax(psd))
    threshold = float(np.max(psd) / 4.0)  # 10*log10(1/4) ~= -6 dB
    below = np.where(psd[peak_idx:] < threshold)[0]
    if len(below) == 0:
        return float(freqs[-1])
    return float(freqs[peak_idx + below[0]])


def plot_example_waveform_psd(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    fs: float,
    index: int,
    save_path: str,
    *,
    title: Optional[str] = None,
    show_minus6db: bool = False,
) -> None:
    """Plot time-domain waveforms and PSD for a single example.

    This keeps the original function name/signature and adds optional title and
    -6 dB annotations.
    """
    noisy = _ensure_2d(noisy)
    denoised = _ensure_2d(denoised)
    clean = _ensure_2d(clean)

    t = np.arange(clean.shape[1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    _plot_waveforms(axes[0], t, clean[index], noisy[index], denoised[index])
    axes[0].set_title(title or f"Temporal domain, index={index}")

    _plot_psd(axes[1], clean[index], noisy[index], denoised[index], fs)
    axes[1].set_title("Frequency domain")

    if show_minus6db:
        clean_f = _minus_6db_frequency(clean[index], fs)
        denoised_f = _minus_6db_frequency(denoised[index], fs)
        axes[1].axvline(clean_f, linestyle="--", alpha=0.8, label=f"Clean -6 dB: {clean_f:.1f} Hz")
        axes[1].axvline(denoised_f, linestyle=":", alpha=0.8, label=f"Denoised -6 dB: {denoised_f:.1f} Hz")
        axes[1].legend(loc="best")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_best_worst_waveform_psd(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    fs: float,
    save_path: str,
    *,
    model_name: str = "Model",
    noise_type: Optional[str] = None,
    show_minus6db: bool = True,
) -> tuple[int, int]:
    """Plot the best and worst samples by temporal RRMSE.

    This is the useful part merged from ``analysis.py``.  It replaces the need
    for a separate best/worst waveform script.

    Returns
    -------
    best_index, worst_index
    """
    noisy = _ensure_2d(noisy)
    denoised = _ensure_2d(denoised)
    clean = _ensure_2d(clean)

    rrmse = _rrmse_per_sample(denoised, clean)
    best_index = int(np.nanargmin(rrmse))
    worst_index = int(np.nanargmax(rrmse))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, (idx, label) in enumerate([(best_index, "Best"), (worst_index, "Worst")]):
        t = np.arange(clean.shape[1]) / fs
        _plot_waveforms(axes[0, col], t, clean[idx], noisy[idx], denoised[idx])
        axes[0, col].set_title(f"{label} sample, index={idx}, RRMSE={rrmse[idx]:.4f}")

        _plot_psd(axes[1, col], clean[idx], noisy[idx], denoised[idx], fs)
        axes[1, col].set_title(f"{label} sample PSD")
        if show_minus6db:
            clean_f = _minus_6db_frequency(clean[idx], fs)
            denoised_f = _minus_6db_frequency(denoised[idx], fs)
            axes[1, col].axvline(clean_f, linestyle="--", alpha=0.8, label=f"Clean -6 dB: {clean_f:.1f} Hz")
            axes[1, col].axvline(denoised_f, linestyle=":", alpha=0.8, label=f"Denoised -6 dB: {denoised_f:.1f} Hz")
            axes[1, col].legend(loc="best", fontsize=8)

    suffix = f" - {noise_type} noise" if noise_type else ""
    fig.suptitle(f"{model_name}{suffix}", fontsize=14, y=1.02)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return best_index, worst_index


def plot_overlay_outputs_waveform_psd(
    noisy: np.ndarray,
    clean: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    fs: float,
    index: int,
    save_path: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Plot one example while overlaying multiple denoising outputs."""
    noisy = _ensure_2d(noisy)
    clean = _ensure_2d(clean)
    outputs = {name: _ensure_2d(arr) for name, arr in outputs.items()}

    t = np.arange(clean.shape[1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    axes[0].plot(t, clean[index], label="Ground-truth EEG", linewidth=1.4)
    axes[0].plot(t, noisy[index], label="Noisy EEG", alpha=0.55, linewidth=1.0)
    for name, arr in outputs.items():
        axes[0].plot(t, arr[index], label=name, alpha=0.9, linewidth=1.0)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title or f"Temporal domain, index={index}")
    axes[0].legend(loc="best")

    f_clean, p_clean = welch(clean[index], fs=fs, nperseg=min(256, clean.shape[1]))
    f_noisy, p_noisy = welch(noisy[index], fs=fs, nperseg=min(256, noisy.shape[1]))
    axes[1].semilogy(f_clean, p_clean, label="Ground-truth EEG", linewidth=1.4)
    axes[1].semilogy(f_noisy, p_noisy, label="Noisy EEG", alpha=0.55, linewidth=1.0)
    for name, arr in outputs.items():
        f, p = welch(arr[index], fs=fs, nperseg=min(256, arr.shape[1]))
        axes[1].semilogy(f, p, label=name, alpha=0.9, linewidth=1.0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Frequency domain")
    axes[1].legend(loc="best")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
