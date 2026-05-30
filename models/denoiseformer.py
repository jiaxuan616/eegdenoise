"""Denoiseformer-style EEG denoising model.

This file keeps the input/output convention of the EEGDNet code you provided:
    input : [B, L], [B, 1, L], or [B, L, 1]
    output: [B, L]

The implementation adds the main structures described in the Denoiseformer paper:
1. slice-based EEG input
2. multiscale feature extraction and fusion
3. transformer encoder / decoder blocks
4. slice pattern attention
5. residual variational autoencoder architecture

Typical use for 2 s EEG at 256 Hz:
    model = Denoiseformer(datanum=512, slice_num=8)
    y_hat, aux = model(noisy, return_aux=True)
    loss, logs = denoiseformer_loss(y_hat, clean, aux, alpha=1.6)  # EOG often 1.6, EMG often 1.4
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _as_2d_signal(x: Tensor, seq_len: int) -> Tensor:
    """Accept [B,L], [B,1,L], or [B,L,1] and return [B,L]."""
    if x.dim() == 3:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.shape[-1] == 1:
            x = x.squeeze(-1)
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    if x.dim() != 2:
        raise ValueError(f"Expected input shape [B, L], got {tuple(x.shape)}")

    if x.shape[-1] != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {x.shape[-1]}")

    return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encoding used by the vanilla transformer."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, dim]

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to [B, N, D]."""
        return x + self.pe[:, : x.shape[1], :].to(dtype=x.dtype)


# ---------------------------------------------------------------------
# Basic transformer blocks
# ---------------------------------------------------------------------


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_mult * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        ff_hidden_mult: int = 4,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, hidden_mult=ff_hidden_mult, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_attn(attn_out)
        x = x + self.ff(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------
# Denoiseformer-specific modules
# ---------------------------------------------------------------------


class MultiscaleFeatureExtractionFusion(nn.Module):
    """Multiscale temporal feature extraction and fusion.

    Input shape:
        [B, M, T], where M is slice_num and T is patch_len.

    Output shape:
        [B, M, D], where D is d_model.

    The paper describes a stack of convolutional feature extraction layers and
    deconvolutional fusion layers. This implementation uses temporal
    ConvTranspose1d layers to progressively upsample the feature dimension
    from patch_len to d_model, while preserving the slice dimension M.
    """

    def __init__(
        self,
        slice_num: int,
        patch_len: int,
        d_model: int = 512,
        levels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")

        self.slice_num = slice_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.levels = levels

        # Use repeated x2 temporal upsampling when possible.
        self.feature_ups = nn.ModuleList()
        self.fusion_ups = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_len = patch_len
        for level in range(levels):
            # For the common 64 -> 128 -> 256 -> 512 setting, stride=2 works exactly.
            # If a different final d_model is used, we interpolate at the end.
            self.feature_ups.append(
                nn.ConvTranspose1d(
                    in_channels=slice_num,
                    out_channels=slice_num,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    groups=1,
                    bias=True,
                )
            )
            self.fusion_ups.append(
                nn.ConvTranspose1d(
                    in_channels=slice_num,
                    out_channels=slice_num,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    groups=1,
                    bias=True,
                )
            )
            in_len *= 2
            self.norms.append(nn.LayerNorm(in_len))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, T] -> [B, M, D]."""
        feature = x
        fused = x

        for feature_up, fusion_up, norm in zip(self.feature_ups, self.fusion_ups, self.norms):
            feature = self.dropout(self.activation(feature_up(feature)))
            fused = fusion_up(fused)

            # Guard against nonstandard lengths.
            if fused.shape[-1] != feature.shape[-1]:
                fused = F.interpolate(fused, size=feature.shape[-1], mode="linear", align_corners=False)

            fused = self.dropout(self.activation(fused + feature))
            fused = norm(fused)

        if fused.shape[-1] != self.d_model:
            fused = F.interpolate(fused, size=self.d_model, mode="linear", align_corners=False)

        return self.final_norm(fused)


class SlicePatternAttention(nn.Module):
    """Slice Pattern Attention.

    Treats EEG slices as channels. It globally pools the temporal feature
    dimension and learns slice-wise gates, then rescales every slice.

    Input/output shape:
        [B, M, D]
    """

    def __init__(self, slice_num: int, reduction: int = 2):
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be >= 1")

        hidden = max(1, slice_num // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Conv1d(slice_num, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, slice_num, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, D]."""
        gates = self.net(self.pool(x))  # [B, M, 1]
        return x * gates


class TransformerEncoder(nn.Module):
    """Transformer-based encoder used in the VAE branch."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        ff_hidden_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    ff_hidden_mult=ff_hidden_mult,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, collect_residuals: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        residual_sum = x
        for layer in self.layers:
            x = layer(x)
            if collect_residuals:
                residual_sum = residual_sum + x

        x = self.norm(x)
        if collect_residuals:
            residual_sum = residual_sum / (len(self.layers) + 1)
            return x, residual_sum
        return x


class TransformerDecoder(nn.Module):
    """Transformer-based decoder with an initial temporal Conv1d layer."""

    def __init__(
        self,
        slice_num: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        ff_hidden_mult: int = 4,
        spa_reduction: int = 2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(slice_num, slice_num, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spa = SlicePatternAttention(slice_num=slice_num, reduction=spa_reduction)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    ff_hidden_mult=ff_hidden_mult,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv(x)
        x = self.spa(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x + residual)


# ---------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------


class Denoiseformer(nn.Module):
    """Denoiseformer-style single-channel EEG artifact removal network.

    Args:
        datanum:
            Input sequence length L. For 2 seconds at 256 Hz, use 512.
        slice_num:
            Number of non-overlapping EEG slices. The paper reports best
            results around 8 slices.
        d_model:
            Transformer feature dimension after multiscale fusion. 512 is a
            natural setting for 2-s 256-Hz segments.
        latent_dim:
            Latent variable dimension. Defaults to d_model.
        enc_depth / dec_depth:
            Number of transformer blocks in encoder and decoder.
        heads:
            Number of attention heads.
        spa_reduction:
            Reduction ratio r in slice pattern attention. The paper reports
            strong results around r=2.
    """

    def __init__(
        self,
        datanum: int,
        slice_num: int = 8,
        d_model: int = 512,
        latent_dim: Optional[int] = None,
        enc_depth: int = 2,
        dec_depth: int = 1,
        heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        multiscale_levels: int = 3,
        spa_reduction: int = 2,
        ff_hidden_mult: int = 4,
    ):
        super().__init__()

        if datanum % slice_num != 0:
            raise ValueError(f"datanum={datanum} must be divisible by slice_num={slice_num}")

        self.seq_len = datanum
        self.slice_num = slice_num
        self.patch_len = datanum // slice_num
        self.d_model = d_model
        self.latent_dim = latent_dim or d_model

        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")

        self.feature_fusion = MultiscaleFeatureExtractionFusion(
            slice_num=slice_num,
            patch_len=self.patch_len,
            d_model=d_model,
            levels=multiscale_levels,
            dropout=dropout,
        )

        self.pos_encoding = SinusoidalPositionalEncoding(max_len=slice_num, dim=d_model)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder_spa = SlicePatternAttention(slice_num=slice_num, reduction=spa_reduction)
        self.encoder = TransformerEncoder(
            dim=d_model,
            depth=enc_depth,
            heads=heads,
            dropout=dropout,
            ff_hidden_mult=ff_hidden_mult,
        )

        self.to_mu = nn.Linear(d_model, self.latent_dim)
        self.to_logvar = nn.Linear(d_model, self.latent_dim)

        self.residual_to_latent = (
            nn.Identity() if self.latent_dim == d_model else nn.Linear(d_model, self.latent_dim)
        )
        self.latent_to_model = (
            nn.Identity() if self.latent_dim == d_model else nn.Linear(self.latent_dim, d_model)
        )

        self.decoder = TransformerDecoder(
            slice_num=slice_num,
            dim=d_model,
            depth=dec_depth,
            heads=heads,
            dropout=dropout,
            ff_hidden_mult=ff_hidden_mult,
            spa_reduction=spa_reduction,
        )

        self.to_patch = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.patch_len),
        )

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, sample: bool = True) -> Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def to_slices(self, x: Tensor) -> Tensor:
        return rearrange(
            x,
            "b (m t) -> b m t",
            m=self.slice_num,
            t=self.patch_len,
        )
        
    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
        sample: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Forward pass.

        Args:
            x:
                [B,L], [B,1,L], or [B,L,1].
            return_aux:
                If True, also return mu/logvar/z for VAE loss.
            sample:
                Whether to sample with the reparameterization trick. Defaults
                to self.training. Use sample=False for deterministic inference.
        """
        x = _as_2d_signal(x, self.seq_len)
        sample = self.training if sample is None else sample

        slices = self.to_slices(x)  # [B, M, T]

        h = self.feature_fusion(slices)     # [B, M, D]
        h = self.pos_encoding(h)
        h = self.emb_dropout(h)

        h = self.encoder_spa(h)
        h, enc_residual = self.encoder(h, collect_residuals=True)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(min=-20.0, max=10.0)

        # Residual variational architecture:
        # z = mu + sigma * eps + residual_information
        z_base = self.reparameterize(mu, logvar, sample=sample)
        z = z_base + self.residual_to_latent(enc_residual)

        dec_in = self.latent_to_model(z)
        dec = self.decoder(dec_in)

        patches = self.to_patch(dec)  # [B, M, T]
        out = rearrange(
            patches,
            "b m t -> b (m t)",
            m=self.slice_num,
            t=self.patch_len,
        )

        if return_aux:
            return out, {
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "z_base": z_base,
                "encoder_residual": enc_residual,
            }

        return out



# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def kl_divergence_standard_normal(mu: Tensor, logvar: Tensor, reduction: str = "mean") -> Tensor:
    """KL[N(mu, sigma^2) || N(0, 1)]."""
    # Shape: [B, M, latent_dim]
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    if reduction == "batchmean":
        return kl.sum() / mu.shape[0]

    raise ValueError(f"Unsupported reduction: {reduction}")


def denoiseformer_loss(
    pred: Tensor,
    target: Tensor,
    aux: Dict[str, Tensor],
    alpha: float = 1.0,
    kl_reduction: str = "mean",
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Total loss: reconstruction loss + alpha * KL loss.

    Paper-style suggestions:
        alpha = 1.6 for EOG artifact removal
        alpha = 1.4 for EMG artifact removal
        alpha = 1.0 can be used as a safe default
    """
    target = _as_2d_signal(target, pred.shape[-1])

    recon_loss = F.mse_loss(pred, target)
    kl_loss = kl_divergence_standard_normal(aux["mu"], aux["logvar"], reduction=kl_reduction)
    total = recon_loss + alpha * kl_loss

    return total, {
        "loss_total": total.detach(),
        "loss_recon": recon_loss.detach(),
        "loss_kl": kl_loss.detach(),
    }


# ---------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    model = Denoiseformer(
        datanum=512,
        slice_num=8,
        d_model=512,
        enc_depth=2,
        dec_depth=1,
        heads=8,
        spa_reduction=2,
    )

    noisy = torch.randn(4, 1, 512)
    clean = torch.randn(4, 512)

    pred, aux = model(noisy, return_aux=True)
    loss, logs = denoiseformer_loss(pred, clean, aux, alpha=1.6)

    print("pred:", tuple(pred.shape))
    print("loss:", float(loss))
    print({k: float(v) for k, v in logs.items()})
