"""Plot training and validation loss curves.

The original ``plot_loss_curve`` API is preserved.  The parser is now more
robust and also supports the vertical, fixed-axis style that previously lived
inside the larger ``analysis.py`` script.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def load_loss_history(result_dir: str, filename: str = "loss_history.npy") -> Mapping[str, Any]:
    """Load a numpy-saved training history dictionary."""
    path = os.path.join(result_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    history = np.load(path, allow_pickle=True).item()
    if not isinstance(history, Mapping):
        raise ValueError(f"Expected a history dictionary in {path}")
    return history


def extract_loss_curves(history: Mapping[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract train/validation MSE curves from common history formats."""
    train_loss = None
    val_loss = None

    loss_dict = history.get("loss") if isinstance(history, Mapping) else None
    if isinstance(loss_dict, Mapping):
        train_loss = loss_dict.get("train_mse", loss_dict.get("train", loss_dict.get("loss")))
        val_loss = loss_dict.get("val_mse", loss_dict.get("val", loss_dict.get("validation")))

    if train_loss is None or val_loss is None:
        for key, value in history.items():
            key_lower = str(key).lower()
            if train_loss is None and ("train" in key_lower or key_lower in {"loss", "mse"}):
                train_loss = value
            if val_loss is None and ("val" in key_lower or "valid" in key_lower):
                val_loss = value

    train_arr = None if train_loss is None else np.asarray(train_loss, dtype=float)
    val_arr = None if val_loss is None else np.asarray(val_loss, dtype=float)
    return train_arr, val_arr


def plot_loss_from_history(
    history: Mapping[str, Any],
    save_path: str,
    *,
    title: str = "Training and validation loss",
    ylabel: str = "MSE",
    figsize: tuple[float, float] = (6, 4),
    ylim: Optional[tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
    show_grid: bool = True,
) -> None:
    """Save a loss-curve figure from an in-memory history dictionary."""
    train_loss, val_loss = extract_loss_curves(history)
    if train_loss is None and val_loss is None:
        raise ValueError("Could not find training or validation loss curves in history.")

    fig, ax = plt.subplots(figsize=figsize)
    if train_loss is not None:
        ax.plot(train_loss, label="Training loss", linewidth=1.2)
    if val_loss is not None:
        ax.plot(val_loss, label="Validation loss", linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    if show_grid:
        ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(
    result_dir: str,
    save_path: str,
    *,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (6, 4),
    ylim: Optional[tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
) -> None:
    """Load loss history and save a training/validation loss plot.

    This keeps the original function name while supporting optional formatting
    that used to be duplicated in ``analysis.py``.
    """
    history = load_loss_history(result_dir)
    plot_loss_from_history(
        history,
        save_path,
        title=title or "Training and validation loss",
        figsize=figsize,
        ylim=ylim,
        y_tick_step=y_tick_step,
    )
    print("Saved:", save_path)


def plot_single_model_loss(
    noise_type: str,
    model_name: str,
    history: Mapping[str, Any],
    save_path: str,
    *,
    figsize: tuple[float, float] = (6, 12),
    ylim: tuple[float, float] = (0, 0.32),
    y_tick_step: float = 0.02,
) -> None:
    """Compatibility wrapper for the vertical loss plots from ``analysis.py``."""
    plot_loss_from_history(
        history,
        save_path,
        title=f"{model_name} - {noise_type}",
        figsize=figsize,
        ylim=ylim,
        y_tick_step=y_tick_step,
    )
