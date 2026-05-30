"""High-level pipeline: data loading, traditional baselines, and deep model training."""

import os

import numpy as np
import torch

from preprocessing.preparation import prepare_data
from models import build_model
from training.optimizer import build_optimizer
from training.trainer import train
from training.save_method import save_eeg
from baselines.filter_baseline import filter_denoise
from baselines.emd_baseline import emd_denoise
from visualization.plot_examples import plot_example_waveform_psd
from visualization.plot_loss import plot_loss_curve
from visualization.plot_examples import plot_example_waveform_psd, load_result
from visualization.plot_metrics import plot_metrics_by_snr



DEEP_MODELS = [
    "fcNN",
    "Simple_CNN",
    "Complex_CNN",
    "RNN_lstm",
    "Novel_CNN",
    "EEGDNet",
    "Denoiseformer",
]


def _get_datanum_fs(noise_type: str):
    """Return (datanum, fs) for a given noise type."""
    if noise_type == "EOG":
        return 512, 256
    elif noise_type == "EMG":
        return 1024, 512
    elif noise_type == "EOG_EMG":
        return 1024, 512
    else:
        raise ValueError("noise_type must be EOG, EMG, or EOG_EMG.")


def _load_data(data_dir: str, noise_type: str):
    """Load raw EEG and noise arrays from disk."""
    EEG_all = np.load(os.path.join(data_dir, "EEG_all_epochs.npy"))

    if noise_type == "EOG":
        noise_all = np.load(os.path.join(data_dir, "EOG_all_epochs.npy"))
    elif noise_type == "EMG":
        noise_all = np.load(os.path.join(data_dir, "EMG_all_epochs.npy"))
    elif noise_type == "EOG_EMG":
        EOG_all = np.load(os.path.join(data_dir, "EOG_all_epochs.npy"))
        EMG_all = np.load(os.path.join(data_dir, "EMG_all_epochs.npy"))
        noise_all = (EOG_all, EMG_all)
    else:
        raise ValueError("noise_type must be EOG, EMG, or EOG_EMG.")

    return EEG_all, noise_all


def run_traditional_baselines(
    noiseEEG_test, EEG_test, test_std_VALUE,
    noise_type, fs,
    result_location, foldername, train_num
):
    """Run filter and EMD baselines, save outputs and figures."""
    output_dir = os.path.join(result_location, foldername, train_num, "nn_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("Running Traditional Methods")
    print("="*50)

    # Filter baseline
    print("Running Filter baseline...")
    filter_output = filter_denoise(
        noisy_eeg=noiseEEG_test,
        noise_type=noise_type,
        fs=fs,
    )
    np.save(os.path.join(output_dir, "Filter_output_test.npy"), filter_output)

    # EMD baseline
    print("Running EMD baseline...")
    emd_output = emd_denoise(
        noisy_eeg=noiseEEG_test,
        noise_type=noise_type,
        fs=fs,
        remove_first_n=1,
    )
    np.save(os.path.join(output_dir, "EMD_output_test.npy"), emd_output)

    # Save ground-truth for plotting
    np.save(os.path.join(output_dir, "EEG_test.npy"), EEG_test)
    np.save(os.path.join(output_dir, "noiseinput_test.npy"), noiseEEG_test)
    np.save(os.path.join(output_dir, "test_std_VALUE.npy"), test_std_VALUE)

    # Figures
    figure_dir = os.path.join(result_location, foldername, train_num, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    idx = 0
    for method_name, denoised_file in [("Filter", "Filter_output_test.npy"), ("EMD", "EMD_output_test.npy")]:
        noisy, denoised, clean = load_result(output_dir, denoised_file)
        plot_example_waveform_psd(
            noisy, denoised, clean, fs, idx,
            os.path.join(figure_dir, f"{method_name}_example_waveform_psd.png"),
        )

    print("Traditional baselines saved to:", output_dir)


def run_deep_model(
    model_name, datanum, fs,
    noiseEEG_train, EEG_train,
    noiseEEG_val, EEG_val,
    noiseEEG_test, EEG_test,
    test_std_VALUE,
    optimizer_name, epochs, batch_size,
    result_location, noise_type,
    train_num="1"
):
    """Train one deep-learning model and produce outputs + figures."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Running Deep Learning Model: {model_name}")
    print(f"Device: {device}")
    print("="*50)

    foldername = f"{noise_type}_{model_name}_{optimizer_name}"
    output_dir = os.path.join(result_location, foldername, train_num, "nn_output")
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(model_name, datanum).to(device)
    optimizer = build_optimizer(optimizer_name, model)

    saved_model, history = train(
        model,
        noiseEEG_train, EEG_train,
        noiseEEG_val, EEG_val,
        epochs, batch_size, optimizer,
        model_name,
        result_location, foldername,
        train_num=train_num,
    )

    save_eeg(
        saved_model,
        result_location, foldername,
        False, False, True,
        noiseEEG_train, EEG_train,
        noiseEEG_val, EEG_val,
        noiseEEG_test, EEG_test,
        train_num=train_num,
    )

    np.save(os.path.join(output_dir, "loss_history.npy"), history)
    np.save(os.path.join(output_dir, "test_std_VALUE.npy"), test_std_VALUE)

    # Figures
    figure_dir = os.path.join(result_location, foldername, train_num, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    noisy, denoised, clean = load_result(output_dir)
    plot_example_waveform_psd(
        noisy, denoised, clean, fs, 0,
        os.path.join(figure_dir, "example_waveform_psd.png"),
    )
    plot_loss_curve(output_dir, os.path.join(figure_dir, "loss_curve.png"))
    plot_metrics_by_snr(
        output_dir, noise_type, model_name,
        os.path.join(figure_dir, "metrics_by_snr.png"),
    )

    print(f"Model {model_name} finished. Output saved to:", output_dir)
    return foldername


def run_full_pipeline(
    data_dir="./data",
    result_location="./results",
    noise_type="EOG",
    epochs=30,
    batch_size=25,
    combin_num=10,
    train_per=0.8,
    optimizer_name="Adam",
    train_num="1",
    deep_models=None,
):
    """Run the full EEG denoising pipeline: traditional baselines + all deep models.

    Parameters
    ----------
    data_dir : str
        Path to directory containing EEG_all_epochs.npy, etc.
    result_location : str
        Path where results will be saved.
    noise_type : str
        One of ``'EOG'``, ``'EMG'``, ``'EOG_EMG'``.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    combin_num : int
        Number of random combinations per segment.
    train_per : float
        Fraction of data used for training.
    optimizer_name : str
        One of ``'Adam'``, ``'RMSprop'``, ``'SGD'``.
    train_num : str
        Run identifier (subfolder name).
    deep_models : list[str] or None
        List of model names to train.  If None, all models in DEEP_MODELS are used.

    Returns
    -------
    dict
        Summary with paths to outputs for each method.
    """
    if deep_models is None:
        deep_models = DEEP_MODELS

    datanum, fs = _get_datanum_fs(noise_type)

    # ---- Load & prepare data ----
    print("Loading data from:", data_dir)
    EEG_all, noise_all = _load_data(data_dir, noise_type)

    print("Preparing train / val / test split ...")
    (
        noiseEEG_train, EEG_train,
        noiseEEG_val, EEG_val,
        noiseEEG_test, EEG_test,
        test_std_VALUE,
    ) = prepare_data(
        EEG_all=EEG_all,
        noise_all=noise_all,
        combin_num=combin_num,
        train_per=train_per,
        noise_type=noise_type,
    )

    summary = {}

    # ---- Traditional baselines ----
    traditional_folder = f"{noise_type}_Traditional"
    run_traditional_baselines(
        noiseEEG_test, EEG_test, test_std_VALUE,
        noise_type, fs,
        result_location, traditional_folder, train_num,
    )
    summary["traditional"] = os.path.join(result_location, traditional_folder, train_num)

    # ---- Deep models ----
    for model_name in deep_models:
        foldername = run_deep_model(
            model_name, datanum, fs,
            noiseEEG_train, EEG_train,
            noiseEEG_val, EEG_val,
            noiseEEG_test, EEG_test,
            test_std_VALUE,
            optimizer_name, epochs, batch_size,
            result_location, noise_type,
            train_num=train_num,
        )
        summary[model_name] = os.path.join(result_location, foldername, train_num)

    print("\n" + "="*50)
    print("All methods finished.")
    print("Results saved under:", result_location)
    print("="*50)

    return summary
