"""Optimiser factory with registry."""

import torch


def _adam(params, lr=5e-5, betas=(0.5, 0.9), eps=1e-8):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)


def _rmsprop(params, lr=5e-5, alpha=0.9):
    return torch.optim.RMSprop(params, lr=lr, alpha=alpha)


def _sgd(params, lr=2e-4, momentum=0.9):
    return torch.optim.SGD(params, lr=lr, momentum=momentum)


OPTIMIZER_REGISTRY = {
    "Adam": _adam,
    "RMSprop": _rmsprop,
    "SGD": _sgd,
}


def build_optimizer(name: str, model: torch.nn.Module):
    """Build an optimiser by name."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name](model.parameters())
