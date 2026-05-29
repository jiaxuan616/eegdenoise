"""Model definitions and registry for EEG denoising networks."""

from .fc_nn import fcNN
from .simple_cnn import simple_CNN
from .complex_cnn import Complex_CNN
from .rnn_lstm import RNN_lstm
from .novel_cnn import Novel_CNN
from .eegdnet import EEGDNet

MODEL_REGISTRY = {
    "fcNN": fcNN,
    "Simple_CNN": simple_CNN,
    "Complex_CNN": Complex_CNN,
    "RNN_lstm": RNN_lstm,
    "Novel_CNN": Novel_CNN,
    "EEGDNet": EEGDNet,
}


def build_model(model_name: str, datanum: int):
    """Factory: build a denoising model by name."""

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](datanum)