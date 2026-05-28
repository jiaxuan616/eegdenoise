"""EMD-based denoising baseline."""

import numpy as np

try:
    from PyEMD import EMD
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False


def _apply_to_batch(data, func):
    data = np.asarray(data)
    if data.ndim == 1:
        return func(data)
    return np.array([func(x) for x in data])


def emd_denoise_one(signal, max_imf=None, remove_first_n=1):
    """EMD denoising approximation.

    The paper used an EMD-based traditional baseline, but the exact IMF
    selection pipeline is not fully reproduced here.

    This version removes the first N IMFs.
    """
    if not HAS_PYEMD:
        raise ImportError(
            "PyEMD is not installed. Install it with: uv pip install EMD-signal"
        )

    emd = EMD()
    imfs = emd(signal, max_imf=max_imf)

    if imfs.size == 0:
        return signal

    if imfs.shape[0] <= remove_first_n:
        return signal

    return np.sum(imfs[remove_first_n:], axis=0)


def emd_denoise(noisy_eeg, noise_type=None, fs=None, max_imf=None, remove_first_n=1):
    """Batch EMD denoising baseline."""
    noisy_eeg = np.asarray(noisy_eeg)
    return _apply_to_batch(
        noisy_eeg,
        lambda x: emd_denoise_one(
            signal=x,
            max_imf=max_imf,
            remove_first_n=remove_first_n,
        ),
    )