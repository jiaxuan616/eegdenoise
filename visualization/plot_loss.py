"""Plot training and validation loss curves."""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(result_dir, save_path):
    """Load loss history and save a training/validation loss plot."""
    history = np.load(
        os.path.join(result_dir, "loss_history.npy"),
        allow_pickle=True,
    ).item()

    train_loss = history["loss"]["train_mse"]
    val_loss = history["loss"]["val_mse"]

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)
