"""EEGdenoiseNet: Deep learning and traditional methods for EEG artifact removal."""

# 核心 pipeline
from .pipeline import run_full_pipeline

# 模型
from .models import MODEL_REGISTRY, build_model

# 训练
from .training import build_optimizer, OPTIMIZER_REGISTRY

# 数据
from .preprocessing import prepare_data

# 传统方法
from .baselines import filter_denoise, emd_denoise

# 可视化
from .visualization import (
    plot_example_waveform_psd,
    plot_loss_curve,
    plot_metrics_by_snr,
)