"""Training loop and helper steps for EEG denoising models."""

import os
import time
import math

import numpy as np
import torch
from tqdm import tqdm

from .loss import denoise_loss_mse


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device)


def _format_input(x, model_name: str):
    """Reshape input for different model architectures.

    fcNN expects  [batch, datanum]
    CNN / LSTM  expect [batch, datanum, 1]
    """
    if model_name == "fcNN":
        return x
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x


def train_step(model, noiseEEG_batch, EEG_batch, optimizer,
               model_name: str, batch_size: int, datanum: int,
               device=None):
    """Single training step: forward + backward."""
    if device is None:
        device = next(model.parameters()).device

    model.train()
    noiseEEG_batch = _to_tensor(noiseEEG_batch, device)
    EEG_batch = _to_tensor(EEG_batch, device)
    noiseEEG_batch = _format_input(noiseEEG_batch, model_name)

    optimizer.zero_grad()
    denoiseoutput = model(noiseEEG_batch)

    if denoiseoutput.dim() == 3:
        denoiseoutput = denoiseoutput.squeeze(-1)
    if EEG_batch.dim() == 3:
        EEG_batch = EEG_batch.squeeze(-1)

    mse_loss = denoise_loss_mse(denoiseoutput, EEG_batch)
    mse_loss.backward()
    optimizer.step()

    first_param = next(model.parameters())
    mse_grads = first_param.grad.detach() if first_param.grad is not None else torch.tensor(0.0, device=device)

    return mse_loss.detach(), mse_grads


@torch.no_grad()
def test_step(model, noiseEEG_test, EEG_test, model_name=None, device=None):
    """Single evaluation step (no grad)."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    noiseEEG_test = _to_tensor(noiseEEG_test, device)
    EEG_test = _to_tensor(EEG_test, device)

    if model_name is not None:
        noiseEEG_test = _format_input(noiseEEG_test, model_name)

    denoiseoutput_test = model(noiseEEG_test)

    if denoiseoutput_test.dim() == 3:
        denoiseoutput_test = denoiseoutput_test.squeeze(-1)
    if EEG_test.dim() == 3:
        EEG_test = EEG_test.squeeze(-1)

    loss = denoise_loss_mse(denoiseoutput_test, EEG_test)
    return denoiseoutput_test.detach().cpu(), loss.detach().cpu()


def train(model, noiseEEG, EEG, noiseEEG_val, EEG_val,
          epochs, batch_size, optimizer, model_name,
          result_location, foldername, train_num,
          loss_callback=None):
    """Full training loop over ``epochs`` epochs.

    Returns
    -------
    saved_model : nn.Module
        Best model (lowest val loss in last 20 % of epochs).
    history : dict
        {"grads": {"mse": ...}, "loss": {"train_mse": ..., "val_mse": ...}}
    """
    device = next(model.parameters()).device

    history = {"grads": {}, "loss": {}}
    train_mse_history, val_mse_history = [], []
    mse_grads_history = []
    val_mse_min = 100.0
    saved_model = model

    train_log_dir = os.path.join(result_location, foldername, train_num, "train")
    val_log_dir = os.path.join(result_location, foldername, train_num, "test")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)

    batch_num = math.ceil(noiseEEG.shape[0] / batch_size)
    datanum = noiseEEG.shape[1]

    for epoch in range(epochs):
        start = time.time()
        mse_grads, train_mse = 0.0, 0.0

        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                noiseEEG_batch = noiseEEG[batch_size * n_batch:batch_size * (n_batch + 1)]
                EEG_batch = EEG[batch_size * n_batch:batch_size * (n_batch + 1)]

                mse_loss_batch, mse_grads_batch = train_step(
                    model, noiseEEG_batch, EEG_batch, optimizer,
                    model_name, batch_size, datanum, device=device,
                )

                mse_grads_batch = torch.sqrt(torch.sum(mse_grads_batch ** 2)).mean().item()
                mse_loss_batch = mse_loss_batch.item()

                train_mse += mse_loss_batch / float(batch_num)
                mse_grads += mse_grads_batch / float(batch_num)
                pbar.update()

        mse_grads_history.append(mse_grads)
        train_mse_history.append(train_mse)

        denoiseoutput, val_mse = test_step(
            model, noiseEEG_val, EEG_val,
            model_name=model_name, device=device,
        )

        val_mse_float = float(val_mse.item())
        val_mse_history.append(val_mse_float)

        if loss_callback is not None:
            loss_callback(epoch + 1, train_mse, val_mse_float)

        # Save best model in last 20 % of training
        if epoch > epochs * 0.8 and val_mse_float < val_mse_min:
            print("yes, smaller", val_mse_float, val_mse_min)
            val_mse_min = val_mse_float
            saved_model = model

            path = os.path.join(result_location, foldername, train_num, "denoise_model")
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, "model_state_dict.pt"))
            torch.save(model, os.path.join(path, "model_full.pt"))
            print("Best model has been saved")

        print(
            "Epoch #: {}/{}, Time taken: {} secs,\n Grads: mse= {},\n Losses: train_mse= {}, val_mse={}".format(
                epoch + 1, epochs, time.time() - start,
                mse_grads, train_mse, val_mse_float,
            )
        )

    history["grads"]["mse"] = mse_grads_history
    history["loss"]["train_mse"] = train_mse_history
    history["loss"]["val_mse"] = val_mse_history

    return saved_model, history
