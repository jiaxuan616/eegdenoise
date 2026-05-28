"""Random data augmentation helpers."""

import numpy as np


def random_signal(signal: np.ndarray, combin_num: int) -> np.ndarray:
    """Create ``combin_num`` shuffled copies of ``signal``.

    Parameters
    ----------
    signal : np.ndarray  (n_segments, timepoints)
    combin_num : int

    Returns
    -------
    np.ndarray  (combin_num, n_segments, timepoints)
    """
    random_result = []
    for _ in range(combin_num):
        perm = np.random.permutation(signal.shape[0])
        shuffled = signal[perm, :]
        random_result.append(shuffled)
    return np.array(random_result)
