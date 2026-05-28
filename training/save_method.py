"""Save denoised outputs and ground-truth as .npy files."""

import os

import numpy as np

from .trainer import test_step


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


def save_eeg(
    saved_model,
    result_location: str,
    foldername: str,
    save_train: bool,
    save_vali: bool,
    save_test: bool,
    noiseEEG_train,
    EEG_train,
    noiseEEG_val,
    EEG_val,
    noiseEEG_test,
    EEG_test,
    train_num: str,
):
    """Save predictions (and optionally inputs / targets) to disk."""
    output_dir = os.path.join(result_location, foldername, train_num, "nn_output")
    os.makedirs(output_dir, exist_ok=True)

    if save_train:
        Denoiseoutput_train, _ = test_step(saved_model, noiseEEG_train, EEG_train)
        np.save(os.path.join(output_dir, "noiseinput_train.npy"), _to_numpy(noiseEEG_train))
        np.save(os.path.join(output_dir, "Denoiseoutput_train.npy"), _to_numpy(Denoiseoutput_train))
        np.save(os.path.join(output_dir, "EEG_train.npy"), _to_numpy(EEG_train))

    if save_vali:
        Denoiseoutput_val, _ = test_step(saved_model, noiseEEG_val, EEG_val)
        np.save(os.path.join(output_dir, "noiseinput_val.npy"), _to_numpy(noiseEEG_val))
        np.save(os.path.join(output_dir, "Denoiseoutput_val.npy"), _to_numpy(Denoiseoutput_val))
        np.save(os.path.join(output_dir, "EEG_val.npy"), _to_numpy(EEG_val))

    if save_test:
        Denoiseoutput_test, _ = test_step(saved_model, noiseEEG_test, EEG_test)
        np.save(os.path.join(output_dir, "noiseinput_test.npy"), _to_numpy(noiseEEG_test))
        np.save(os.path.join(output_dir, "Denoiseoutput_test.npy"), _to_numpy(Denoiseoutput_test))
        np.save(os.path.join(output_dir, "EEG_test.npy"), _to_numpy(EEG_test))
