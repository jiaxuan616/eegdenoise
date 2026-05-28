"""Traditional baselines: Butterworth filter and EMD-based denoising."""

from .filter_baseline import filter_denoise, preprocess_clean_eeg_segments, preprocess_eog_segments, preprocess_emg_segments
from .emd_baseline import emd_denoise, emd_denoise_one
