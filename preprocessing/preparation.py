"""Train/val/test splitting, SNR mixing, and standardisation for EEG denoising."""

import numpy as np
from scipy.signal import resample

from .augment import random_signal

def _get_rms(records):
    return np.sqrt(np.mean(records ** 2))


def _upsample_to_1024(data: np.ndarray, name: str) -> np.ndarray:
    """Upsample from 512 → 1024 time points if needed."""
    if data.shape[1] == 1024:
        return data
    if data.shape[1] == 512:
        print(f"Upsampling {name} from 512 to 1024 points.")
        return resample(data, 1024, axis=1)
    raise ValueError(
        f"{name} shape error: expected 512 or 1024 time points, got {data.shape[1]}"
    )


def prepare_data(
    EEG_all: np.ndarray,
    noise_all,
    combin_num: int,
    train_per: float,
    noise_type: str,
):
    """Prepare train / val / test sets for EEG denoising.

    Parameters
    ----------
    EEG_all : np.ndarray  (n_segments, timepoints)
        Clean EEG epochs.
    noise_all : np.ndarray or tuple of np.ndarray
        Single noise source (EOG or EMG), or tuple (EOG_all, EMG_all) for EOG_EMG.
    combin_num : int
        Number of random combinations per segment for data augmentation.
    train_per : float
        Fraction of data used for training (0–1).
    noise_type : str
        One of ``'EOG'``, ``'EMG'``, ``'EOG_EMG'``.

    Returns
    -------
    tuple : (noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
             noiseEEG_test, EEG_test, test_std_VALUE)
        All arrays are standardised (divided by per-sample std).
    """
    noise_type = noise_type.upper()

    # ---- up-sample EMG / EOG_EMG to 1024 Hz equivalent ----
    if noise_type == "EMG":
        EEG_all = _upsample_to_1024(EEG_all, "EEG")
        noise_all = _upsample_to_1024(noise_all, "EMG")
    elif noise_type == "EOG_EMG":
        EOG_all, EMG_all = noise_all
        EEG_all = _upsample_to_1024(EEG_all, "EEG")
        EOG_all = _upsample_to_1024(EOG_all, "EOG")
        EMG_all = _upsample_to_1024(EMG_all, "EMG")
        noise_all = (EOG_all, EMG_all)

    # ---- shuffle & align segment counts ----
    EEG_all_random = np.squeeze(random_signal(EEG_all, combin_num=1))

    if noise_type == "EOG_EMG":
        EOG_all, EMG_all = noise_all
        EOG_all_random = np.squeeze(random_signal(EOG_all, combin_num=1))
        EMG_all_random = np.squeeze(random_signal(EMG_all, combin_num=1))
        min_num = min(EEG_all_random.shape[0], EOG_all_random.shape[0], EMG_all_random.shape[0])
        EEG_all_random = EEG_all_random[:min_num]
        EOG_all_random = EOG_all_random[:min_num]
        EMG_all_random = EMG_all_random[:min_num]
        print("EEG/EOG/EMG segments after alignment:", min_num)
    else:
        noise_all_random = np.squeeze(random_signal(noise_all, combin_num=1))
        if noise_type == "EMG":
            reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
            if reuse_num > 0:
                EEG_all_random = np.vstack([EEG_all_random[:reuse_num], EEG_all_random])
                print("EEG segments after reuse:", EEG_all_random.shape[0])
            elif reuse_num < 0:
                EEG_all_random = EEG_all_random[:noise_all_random.shape[0]]
                print("EEG segments after drop:", EEG_all_random.shape[0])
            else:
                print("EEG/EMG segments already aligned:", EEG_all_random.shape[0])
        else:  # EOG
            EEG_all_random = EEG_all_random[:noise_all_random.shape[0]]
            print("EEG segments after drop:", EEG_all_random.shape[0])

    timepoint = EEG_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    val_num = round((EEG_all_random.shape[0] - train_num) / 2)

    train_eeg = EEG_all_random[:train_num]
    val_eeg = EEG_all_random[train_num:train_num + val_num]
    test_eeg = EEG_all_random[train_num + val_num:]

    # extract noise partitions
    if noise_type == "EOG_EMG":
        train_eog = EOG_all_random[:train_num]
        val_eog = EOG_all_random[train_num:train_num + val_num]
        test_eog = EOG_all_random[train_num + val_num:]
        train_emg = EMG_all_random[:train_num]
        val_emg = EMG_all_random[train_num:train_num + val_num]
        test_emg = EMG_all_random[train_num + val_num:]
    else:
        train_noise = noise_all_random[:train_num]
        val_noise = noise_all_random[train_num:train_num + val_num]
        test_noise = noise_all_random[train_num + val_num:]

    # ============================
    #  Build training set
    # ============================
    EEG_train = random_signal(train_eeg, combin_num=combin_num).reshape(-1, timepoint)

    if noise_type == "EOG_EMG":
        EOG_train = random_signal(train_eog, combin_num=combin_num).reshape(-1, timepoint)
        EMG_train = random_signal(train_emg, combin_num=combin_num).reshape(-1, timepoint)

        snr_eog_dB = np.random.uniform(-7, 2, EEG_train.shape[0])
        snr_emg_dB = np.random.uniform(-7, 2, EEG_train.shape[0])
        snr_eog = 10 ** (0.1 * snr_eog_dB)
        snr_emg = 10 ** (0.1 * snr_emg_dB)

        noiseEEG_train = np.empty_like(EEG_train)
        for i in range(EEG_train.shape[0]):
            coe_eog = _get_rms(EEG_train[i]) / (_get_rms(EOG_train[i]) * snr_eog[i])
            coe_emg = _get_rms(EEG_train[i]) / (_get_rms(EMG_train[i]) * snr_emg[i])
            noiseEEG_train[i] = EEG_train[i] + coe_eog * EOG_train[i] + coe_emg * EMG_train[i]
    else:
        NOISE_train = random_signal(train_noise, combin_num=combin_num).reshape(-1, timepoint)
        snr_dB = np.random.uniform(-7, 2, EEG_train.shape[0])
        snr = 10 ** (0.1 * snr_dB)

        noiseEEG_train = np.empty_like(EEG_train)
        for i in range(EEG_train.shape[0]):
            coe = _get_rms(EEG_train[i]) / (_get_rms(NOISE_train[i]) * snr[i])
            noiseEEG_train[i] = EEG_train[i] + coe * NOISE_train[i]

    # standardise training set
    std_train = np.std(noiseEEG_train, axis=1, keepdims=True)
    noiseEEG_train = noiseEEG_train / std_train
    EEG_train = EEG_train / std_train
    print("training data prepared", noiseEEG_train.shape, EEG_train.shape)

    # ============================
    #  Build validation & test sets
    # ============================
    def _build_fixed_snr_set(eeg_seg, noise_parts, snr_dB_values):
        eeg_list, noisy_list = [], []
        for sdB in snr_dB_values:
            s_val = 10 ** (0.1 * sdB)
            for j in range(eeg_seg.shape[0]):
                eeg = eeg_seg[j]
                if noise_type == "EOG_EMG":
                    eog, emg = noise_parts
                    coe_eog = _get_rms(eeg) / (_get_rms(eog[j]) * s_val)
                    coe_emg = _get_rms(eeg) / (_get_rms(emg[j]) * s_val)
                    neeg = eeg + coe_eog * eog[j] + coe_emg * emg[j]
                else:
                    noi = noise_parts[j]
                    coe = _get_rms(eeg) / (_get_rms(noi) * s_val)
                    neeg = eeg + coe * noi
                eeg_list.append(eeg)
                noisy_list.append(neeg)
        return np.array(noisy_list), np.array(eeg_list)

    snr_range = np.linspace(-7.0, 2.0, num=10)

    if noise_type == "EOG_EMG":
        noiseEEG_val, EEG_val = _build_fixed_snr_set(val_eeg, (val_eog, val_emg), snr_range)
        noiseEEG_test, EEG_test = _build_fixed_snr_set(test_eeg, (test_eog, test_emg), snr_range)
    else:
        noiseEEG_val, EEG_val = _build_fixed_snr_set(val_eeg, val_noise, snr_range)
        noiseEEG_test, EEG_test = _build_fixed_snr_set(test_eeg, test_noise, snr_range)

    # standardise validation
    std_val = np.std(noiseEEG_val, axis=1, keepdims=True)
    noiseEEG_val = noiseEEG_val / std_val
    EEG_val = EEG_val / std_val
    print("validation data prepared", noiseEEG_val.shape, EEG_val.shape)

    # standardise test  (keep original std for later re-scaling)
    std_VALUE = np.std(noiseEEG_test, axis=1)
    noiseEEG_test = noiseEEG_test / std_VALUE[:, np.newaxis]
    EEG_test = EEG_test / std_VALUE[:, np.newaxis]
    print("test data prepared", noiseEEG_test.shape, EEG_test.shape)

    return (
        noiseEEG_train,
        EEG_train,
        noiseEEG_val,
        EEG_val,
        noiseEEG_test,
        EEG_test,
        std_VALUE,
    )