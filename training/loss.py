"""Loss functions for EEG denoising."""

import torch
import torch.nn.functional as F


def denoise_loss_mse(denoise: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(denoise, clean)


def denoise_loss_rmse(denoise: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Root mean squared error loss."""
    return torch.sqrt(F.mse_loss(denoise, clean))


def denoise_loss_rrmset(denoise: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Relative root mean squared error (temporal)."""
    rmse1 = denoise_loss_rmse(denoise, clean)
    rmse2 = denoise_loss_rmse(clean, torch.zeros_like(clean))
    return rmse1 / rmse2
