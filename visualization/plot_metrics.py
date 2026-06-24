"""Metrics and metric plots for EEG denoising results.

This file keeps the original SNR-based RRMSE/CC helpers and consolidates the
non-duplicated logic from the separate metric scripts:

- advanced metrics: LSD, gamma energy ratio, envelope correlation, SI-SNR, SSIM;
- band metrics: energy ratio, band RRMSE, band Pearson correlation, band LSD,
  spectral slope change, centroid shift, and flatness;
- clean-reference metrics and comparison plots.

The old standalone ``compute_band_energy_ratios.py`` is intentionally not
recreated because its Energy Ratio is already included in band metrics.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert, stft, welch

warnings.filterwarnings("ignore")

FS_MAP = {"EOG": 256, "EMG": 512, "EOG_EMG": 512}
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 50),
}


def get_fs(noise_type: str, default: int = 512) -> int:
    """Return the sampling rate used by the EEGdenoiseNet noise type."""
    return FS_MAP.get(noise_type, default)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def _rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def _band_energy(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    idx = _band_mask(freqs, band)
    if not np.any(idx):
        return np.nan
    return float(np.trapz(psd[idx], freqs[idx]))


def spectral_flatness(psd: np.ndarray, eps: float = 1e-12) -> float:
    """Compute spectral flatness as geometric mean / arithmetic mean."""
    psd = np.asarray(psd, dtype=float)
    if psd.size == 0:
        return np.nan
    gm = np.exp(np.mean(np.log(psd + eps)))
    am = np.mean(psd + eps)
    return float(gm / am)


# ---------------------------------------------------------------------------
# Original SNR-grouped metrics API
# ---------------------------------------------------------------------------

def _rrmse_temporal(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    pred = _ensure_2d(pred)
    clean = _ensure_2d(clean)
    return _rms(pred - clean, axis=1) / (_rms(clean, axis=1) + 1e-12)


def _rrmse_spectral(pred: np.ndarray, clean: np.ndarray, fs: float) -> np.ndarray:
    pred = _ensure_2d(pred)
    clean = _ensure_2d(clean)
    vals = []
    for p, c in zip(pred, clean):
        _, psd_p = welch(p, fs=fs, nperseg=min(256, len(p)))
        _, psd_c = welch(c, fs=fs, nperseg=min(256, len(c)))
        vals.append(float(_rms(psd_p - psd_c) / (_rms(psd_c) + 1e-12)))
    return np.array(vals)


def _correlation_coeff(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    pred = _ensure_2d(pred)
    clean = _ensure_2d(clean)
    return np.array([_safe_corr(p, c) for p, c in zip(pred, clean)])


def _group_by_snr(values: np.ndarray, snr_num: int = 10) -> list[np.ndarray]:
    return np.array_split(np.asarray(values), snr_num)


def compute_metrics(pred: np.ndarray, clean: np.ndarray, fs: float, snr_num: int = 10) -> dict[str, np.ndarray]:
    """Compute RRMSE and correlation coefficient per SNR group."""
    rt = _rrmse_temporal(pred, clean)
    rs = _rrmse_spectral(pred, clean, fs)
    cc = _correlation_coeff(pred, clean)
    return {
        "rrmse_temporal": np.array([np.nanmean(x) for x in _group_by_snr(rt, snr_num)]),
        "rrmse_spectral": np.array([np.nanmean(x) for x in _group_by_snr(rs, snr_num)]),
        "cc": np.array([np.nanmean(x) for x in _group_by_snr(cc, snr_num)]),
    }


def plot_metrics_by_snr(
    result_dir: str,
    noise_type: str,
    label: str,
    save_path: str,
    *,
    denoised_file: str = "Denoiseoutput_test.npy",
    snrs: Optional[Sequence[float]] = None,
) -> None:
    """Plot three metric panels versus SNR for one denoising method."""
    fs = get_fs(noise_type)
    clean = _ensure_2d(np.load(os.path.join(result_dir, "EEG_test.npy"), allow_pickle=True))
    denoised = _ensure_2d(np.load(os.path.join(result_dir, denoised_file), allow_pickle=True))
    metrics = compute_metrics(denoised, clean, fs)
    snrs = np.asarray(snrs if snrs is not None else np.arange(-7, 3))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("rrmse_temporal", "RRMSE temporal", "Temporal RRMSE"),
        ("rrmse_spectral", "RRMSE spectral", "Spectral RRMSE"),
        ("cc", "CC", "Correlation coefficient"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.plot(snrs, metrics[key], marker="o", label=label)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    np.save(os.path.join(result_dir, "metrics_by_snr.npy"), metrics)
    print("Saved:", save_path)


def plot_multi_metrics_by_snr(
    result_dir: str,
    noise_type: str,
    output_dict: Mapping[str, np.ndarray],
    save_path: str,
    *,
    snrs: Optional[Sequence[float]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Plot SNR-grouped metrics for multiple denoising methods."""
    fs = get_fs(noise_type)
    clean = _ensure_2d(np.load(os.path.join(result_dir, "EEG_test.npy"), allow_pickle=True))
    snrs = np.asarray(snrs if snrs is not None else np.arange(-7, 3))

    all_metrics = {name: compute_metrics(pred, clean, fs) for name, pred in output_dict.items()}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("rrmse_temporal", "RRMSE temporal", "Temporal RRMSE"),
        ("rrmse_spectral", "RRMSE spectral", "Spectral RRMSE"),
        ("cc", "CC", "Correlation coefficient"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name, metrics in all_metrics.items():
            ax.plot(snrs, metrics[key], marker="o", label=name)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    np.save(os.path.join(result_dir, "metrics_by_snr.npy"), all_metrics)
    print("Saved:", save_path)
    return all_metrics


# ---------------------------------------------------------------------------
# Advanced sample-level metrics merged from compute_metrics.py
# ---------------------------------------------------------------------------

def log_spectral_distance(clean: np.ndarray, denoised: np.ndarray, fs: float, nperseg: int = 256) -> float:
    """Log-spectral distance in dB; lower is better."""
    f, psd_clean = welch(clean, fs=fs, nperseg=min(nperseg, len(clean)), return_onesided=True)
    _, psd_denoised = welch(denoised, fs=fs, nperseg=min(nperseg, len(denoised)), return_onesided=True)
    eps = 1e-10
    return float(np.sqrt(np.mean((10 * np.log10((psd_clean + eps) / (psd_denoised + eps))) ** 2)))


def band_energy_ratio(clean: np.ndarray, denoised: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Energy ratio E_denoised / E_clean for a frequency band; ideal is 1."""
    f, psd_clean = welch(clean, fs=fs, nperseg=min(256, len(clean)), return_onesided=True)
    _, psd_denoised = welch(denoised, fs=fs, nperseg=min(256, len(denoised)), return_onesided=True)
    e_clean = _band_energy(f, psd_clean, band)
    e_denoised = _band_energy(f, psd_denoised, band)
    if not np.isfinite(e_clean) or e_clean < 1e-10:
        return 1.0
    return float(e_denoised / e_clean)


def envelope_correlation(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Correlation between Hilbert envelopes; higher is better."""
    return _safe_corr(np.abs(hilbert(clean)), np.abs(hilbert(denoised)))


def si_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Scale-invariant SNR in dB; higher is better."""
    clean = np.asarray(clean).flatten()
    denoised = np.asarray(denoised).flatten()
    alpha = np.dot(clean, denoised) / (np.dot(denoised, denoised) + 1e-10)
    s_target = alpha * denoised
    e_noise = clean - s_target
    return float(10 * np.log10(np.dot(clean, clean) / (np.dot(e_noise, e_noise) + 1e-10)))


def spectrogram_ssim(clean: np.ndarray, denoised: np.ndarray, fs: float, nperseg: int = 128, noverlap: int = 64) -> float:
    """SSIM between normalized STFT magnitude spectrograms; higher is better."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError as exc:
        raise ImportError("spectrogram_ssim requires scikit-image. Install skimage or set compute_ssim=False.") from exc

    _, _, z_clean = stft(clean, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, z_denoised = stft(denoised, fs, nperseg=nperseg, noverlap=noverlap)
    mag_clean = np.abs(z_clean)
    mag_denoised = np.abs(z_denoised)
    max_val = max(float(mag_clean.max()), float(mag_denoised.max()), 1e-12)
    mag_clean = mag_clean / max_val
    mag_denoised = mag_denoised / max_val

    min_dim = min(mag_clean.shape)
    if min_dim < 3:
        return np.nan
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    return float(ssim(mag_clean, mag_denoised, data_range=1.0, win_size=win_size))


def compute_advanced_metrics(
    clean_batch: np.ndarray,
    denoised_batch: np.ndarray,
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    gamma_band_name: str = "Gamma",
    compute_ssim: bool = True,
) -> pd.DataFrame:
    """Compute LSD, Gamma ratio, envelope correlation, SI-SNR, and optional SSIM.

    Returns one row per sample.
    """
    clean_batch = _ensure_2d(clean_batch)
    denoised_batch = _ensure_2d(denoised_batch)
    gamma_band = bands[gamma_band_name]

    rows = []
    for clean, denoised in zip(clean_batch, denoised_batch):
        row = {
            "LSD": log_spectral_distance(clean, denoised, fs),
            "Gamma_ratio": band_energy_ratio(clean, denoised, fs, gamma_band),
            "Env_corr": envelope_correlation(clean, denoised),
            "SI_SNR": si_snr(clean, denoised),
        }
        if compute_ssim:
            row["SSIM"] = spectrogram_ssim(clean, denoised, fs)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_dataframe(df_metrics: pd.DataFrame, *, model_name: Optional[str] = None) -> dict[str, float | str]:
    """Summarize per-sample metrics as mean/std columns."""
    summary: dict[str, float | str] = {}
    if model_name is not None:
        summary["Model"] = model_name
    for col in df_metrics.columns:
        vals = pd.to_numeric(df_metrics[col], errors="coerce")
        summary[f"{col}_mean"] = float(vals.mean())
        summary[f"{col}_std"] = float(vals.std())
    return summary


# ---------------------------------------------------------------------------
# Band metrics merged from comprehensive_band_analysis.py and compute_band_lsd.py
# ---------------------------------------------------------------------------

def compute_band_metrics_single(
    clean: np.ndarray,
    denoised: np.ndarray,
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
) -> dict[str, dict[str, float]]:
    """Compute all useful per-band metrics for one sample."""
    f, psd_clean = welch(clean, fs=fs, nperseg=min(256, len(clean)), return_onesided=True)
    _, psd_denoised = welch(denoised, fs=fs, nperseg=min(256, len(denoised)), return_onesided=True)

    results: dict[str, dict[str, float]] = {}
    eps = 1e-10
    for band_name, band in bands.items():
        idx = _band_mask(f, band)
        if not np.any(idx):
            results[band_name] = {
                "Energy_Ratio": np.nan,
                "Band_RRMSE": np.nan,
                "Band_Pearson": np.nan,
                "Band_LSD": np.nan,
                "Slope_Change": np.nan,
                "Centroid_Shift": np.nan,
                "Flatness": np.nan,
            }
            continue

        freqs = f[idx]
        pc = psd_clean[idx]
        pd_ = psd_denoised[idx]
        e_clean = float(np.trapz(pc, freqs))
        e_denoised = float(np.trapz(pd_, freqs))
        energy_ratio = e_denoised / (e_clean + eps) if e_clean > eps else 1.0
        band_rrmse = float(_rms(pd_ - pc) / (_rms(pc) + eps))
        band_pearson = _safe_corr(pc, pd_)
        band_lsd = float(np.sqrt(np.mean((10 * np.log10((pc + eps) / (pd_ + eps))) ** 2)))

        if len(freqs) >= 2:
            log_f = np.log10(freqs + eps)
            slope_clean = np.polyfit(log_f, np.log10(pc + eps), 1)[0]
            slope_denoised = np.polyfit(log_f, np.log10(pd_ + eps), 1)[0]
            slope_change = float(slope_denoised - slope_clean)
        else:
            slope_change = np.nan

        centroid_clean = float(np.sum(freqs * pc) / (np.sum(pc) + eps))
        centroid_denoised = float(np.sum(freqs * pd_) / (np.sum(pd_) + eps))

        results[band_name] = {
            "Energy_Ratio": float(energy_ratio),
            "Band_RRMSE": band_rrmse,
            "Band_Pearson": band_pearson,
            "Band_LSD": band_lsd,
            "Slope_Change": slope_change,
            "Centroid_Shift": float(centroid_denoised - centroid_clean),
            "Flatness": spectral_flatness(pd_),
        }
    return results


def compute_band_metrics_batch(
    clean_batch: np.ndarray,
    denoised_batch: np.ndarray,
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    model_name: Optional[str] = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Compute band metrics for a batch.

    If ``aggregate=True``, returns one row per band with means. Otherwise,
    returns one row per sample and band.
    """
    clean_batch = _ensure_2d(clean_batch)
    denoised_batch = _ensure_2d(denoised_batch)

    rows = []
    for sample_index, (clean, denoised) in enumerate(zip(clean_batch, denoised_batch)):
        sample_metrics = compute_band_metrics_single(clean, denoised, fs, bands=bands)
        for band_name, metrics in sample_metrics.items():
            row = {"Sample": sample_index, "Band": band_name, **metrics}
            if model_name is not None:
                row["Model"] = model_name
            rows.append(row)
    df = pd.DataFrame(rows)
    if not aggregate:
        return df

    group_cols = ["Band"] if model_name is None else ["Model", "Band"]
    value_cols = [c for c in df.columns if c not in {"Sample", "Model", "Band"}]
    return df.groupby(group_cols, as_index=False)[value_cols].mean(numeric_only=True)


def compute_signal_flatness(
    signal: np.ndarray,
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    overall_band: tuple[float, float] = (1, 50),
) -> dict[str, float]:
    """Compute spectral flatness by band plus overall flatness."""
    f, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)), return_onesided=True)
    out = {}
    for band_name, band in bands.items():
        out[band_name] = spectral_flatness(psd[_band_mask(f, band)])
    out["Overall"] = spectral_flatness(psd[_band_mask(f, overall_band)])
    return out


def compute_flatness_summary(
    signal_dict: Mapping[str, np.ndarray],
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
) -> pd.DataFrame:
    """Compute mean spectral flatness for Raw and/or denoised method arrays."""
    rows = []
    for name, batch in signal_dict.items():
        batch = _ensure_2d(batch)
        accum = {band: [] for band in list(bands.keys()) + ["Overall"]}
        for sig in batch:
            vals = compute_signal_flatness(sig, fs, bands=bands)
            for key, val in vals.items():
                accum[key].append(val)
        row = {"Model": name, **{key: float(np.nanmean(vals)) for key, vals in accum.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


def compute_clean_reference_metrics(
    clean_batch: np.ndarray,
    fs: float,
    *,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    overall_band: tuple[float, float] = (1, 50),
) -> pd.DataFrame:
    """Compute clean EEG reference band proportions and overall flatness."""
    clean_batch = _ensure_2d(clean_batch)
    rows = []
    for i, sig in enumerate(clean_batch):
        f, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)), return_onesided=True)
        total_energy = _band_energy(f, psd, overall_band)
        row = {"epoch": i, "Overall_flatness": spectral_flatness(psd[_band_mask(f, overall_band)])}
        for band_name, band in bands.items():
            energy = _band_energy(f, psd, band)
            row[f"{band_name}_prop"] = float(energy / (total_energy + 1e-12) * 100)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Directory-level helpers that replace standalone scripts
# ---------------------------------------------------------------------------

def collect_result_dirs(
    results_dir: str,
    *,
    noise_type: str = "EOG_EMG",
    run_id: str = "1",
    include_traditional: bool = True,
) -> dict[str, str]:
    """Collect model output directories from an EEGdenoiseNet results folder."""
    methods: dict[str, str] = {}
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for folder in os.listdir(results_dir):
        if not folder.startswith(f"{noise_type}_") or not folder.endswith("_Adam"):
            continue
        model_name = folder[len(noise_type) + 1 : -5]
        model_path = os.path.join(results_dir, folder, run_id, "nn_output")
        if os.path.exists(model_path):
            methods[model_name] = model_path

    if include_traditional:
        trad_path = os.path.join(results_dir, f"{noise_type}_Traditional", run_id, "nn_output")
        if os.path.exists(trad_path):
            if os.path.exists(os.path.join(trad_path, "Filter_output_test.npy")):
                methods["Filter"] = trad_path
            if os.path.exists(os.path.join(trad_path, "EMD_output_test.npy")):
                methods["EMD"] = trad_path
    return methods


def load_clean_denoised(
    result_dir: str,
    *,
    method_name: Optional[str] = None,
    denoised_file: Optional[str] = None,
    apply_std: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load clean and denoised arrays for a neural or traditional method."""
    clean = _ensure_2d(np.load(os.path.join(result_dir, "EEG_test.npy"), allow_pickle=True))
    if denoised_file is None:
        if method_name in {"Filter", "EMD"}:
            denoised_file = f"{method_name}_output_test.npy"
        else:
            denoised_file = "Denoiseoutput_test.npy"
    denoised = _ensure_2d(np.load(os.path.join(result_dir, denoised_file), allow_pickle=True))

    if apply_std:
        std_path = os.path.join(result_dir, "test_std_VALUE.npy")
        if os.path.exists(std_path):
            std = np.load(std_path, allow_pickle=True).reshape(-1, 1)
            clean = clean * std
            denoised = denoised * std
    return clean, denoised


def summarize_advanced_metrics_for_results(
    results_dir: str,
    *,
    noise_type: str = "EOG_EMG",
    run_id: str = "1",
    output_csv: Optional[str] = None,
    compute_ssim: bool = True,
) -> pd.DataFrame:
    """Create the former ``summary_table.csv`` advanced-metric report."""
    fs = get_fs(noise_type)
    methods = collect_result_dirs(results_dir, noise_type=noise_type, run_id=run_id)
    rows = []
    for model, result_dir in methods.items():
        clean, denoised = load_clean_denoised(result_dir, method_name=model)
        df_metrics = compute_advanced_metrics(clean, denoised, fs, compute_ssim=compute_ssim)
        rows.append(summarize_metric_dataframe(df_metrics, model_name=model))
    out = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def summarize_band_metrics_for_results(
    results_dir: str,
    *,
    noise_type: str = "EOG_EMG",
    run_id: str = "1",
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Create the merged comprehensive band report.

    This replaces ``comprehensive_band_analysis.py``, ``compute_band_lsd.py``,
    and the duplicated ``compute_band_energy_ratios.py``.
    """
    fs = get_fs(noise_type)
    methods = collect_result_dirs(results_dir, noise_type=noise_type, run_id=run_id)
    frames = []
    for model, result_dir in methods.items():
        clean, denoised = load_clean_denoised(result_dir, method_name=model)
        frames.append(compute_band_metrics_batch(clean, denoised, fs, model_name=model, aggregate=True))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


# ---------------------------------------------------------------------------
# Plot helpers merged from analysis/compare scripts
# ---------------------------------------------------------------------------

def plot_advanced_metric_bars(
    summary_df: pd.DataFrame,
    metric: str,
    save_path: str,
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Plot mean/std bars for one advanced metric from summary_table-style data."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns:
        raise KeyError(f"Missing column: {mean_col}")

    models = summary_df["Model"].astype(str).to_numpy()
    means = summary_df[mean_col].astype(float).to_numpy()
    stds = summary_df[std_col].astype(float).to_numpy() if std_col in summary_df.columns else None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5 if stds is not None else 0, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_band_metric_bars(
    band_df: pd.DataFrame,
    metric: str,
    save_path: str,
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Plot a band metric grouped by model and frequency band."""
    if not {"Model", "Band", metric}.issubset(band_df.columns):
        raise KeyError(f"band_df must contain Model, Band, and {metric}")

    models = list(band_df["Model"].drop_duplicates())
    bands = list(band_df["Band"].drop_duplicates())
    x = np.arange(len(models))
    width = min(0.8 / max(len(bands), 1), 0.16)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, band in enumerate(bands):
        vals = (
            band_df[band_df["Band"] == band]
            .set_index("Model")
            .reindex(models)[metric]
            .astype(float)
            .to_numpy()
        )
        ax.bar(x + i * width, vals, width, label=band)

    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by frequency band")
    ax.set_xticks(x + width * (len(bands) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Band")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_flatness_summary(
    flatness_df: pd.DataFrame,
    save_path: str,
    *,
    clean_reference_df: Optional[pd.DataFrame] = None,
) -> None:
    """Plot overall spectral flatness, optionally with a clean EEG reference band."""
    if not {"Model", "Overall"}.issubset(flatness_df.columns):
        raise KeyError("flatness_df must contain Model and Overall columns")

    models = flatness_df["Model"].astype(str).to_list()
    vals = flatness_df["Overall"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(models, vals, alpha=0.8, label="Models / Raw")
    if clean_reference_df is not None and "Overall_flatness" in clean_reference_df.columns:
        clean_vals = clean_reference_df["Overall_flatness"].astype(float).to_numpy()
        mean = float(np.nanmean(clean_vals))
        std = float(np.nanstd(clean_vals))
        ax.axhline(mean, linestyle="-", linewidth=2, label=f"Clean mean ({mean:.3f})")
        ax.fill_between([-0.5, len(models) - 0.5], mean - std, mean + std, alpha=0.15, label=f"Clean ±1σ ({std:.3f})")

    ax.set_ylabel("Spectral flatness")
    ax.set_title("Spectral flatness comparison")
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_clean_band_proportion_comparison(
    clean_reference_df: pd.DataFrame,
    proportion_df: pd.DataFrame,
    save_path: str,
    *,
    title: str = "Band energy proportion: methods vs clean EEG",
) -> None:
    """Plot method band proportions with clean EEG mean markers.

    This consolidates the useful part of ``compare_band_proportions.py``.
    """
    prop_cols = [f"{band}_prop" for band in BANDS]
    missing = [col for col in prop_cols if col not in clean_reference_df.columns]
    if missing:
        raise KeyError(f"clean_reference_df is missing columns: {missing}")

    clean_avg = clean_reference_df[prop_cols].mean().to_numpy()
    if "Model" in proportion_df.columns:
        proportion_df = proportion_df.set_index("Model")
    bands = list(proportion_df.columns)
    methods = list(proportion_df.index)

    x = np.arange(len(bands))
    width = min(0.8 / max(len(methods), 1), 0.08)
    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(methods)), 6))

    for i, method in enumerate(methods):
        vals = proportion_df.loc[method].astype(float).to_numpy()
        ax.bar(x + i * width, vals, width, label=method)

    for i, _band in enumerate(bands):
        ax.scatter(
            x[i] + np.arange(len(methods)) * width,
            [clean_avg[i]] * len(methods),
            marker="_",
            s=120,
            linewidth=2,
            label="Clean EEG mean" if i == 0 else "",
        )

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Energy proportion (%)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
