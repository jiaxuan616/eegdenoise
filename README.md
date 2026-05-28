# EEGdenoiseNet

EEGdenoiseNet 是一个用于 EEG 信号去噪的实验项目，支持使用深度学习模型和传统信号处理方法对含噪 EEG 信号进行去噪对比实验。

项目目前支持三类噪声场景：

* `EOG`：眼电伪迹去噪
* `EMG`：肌电伪迹去噪
* `EOG_EMG`：混合眼电与肌电伪迹去噪

支持的深度学习模型包括：

* `fcNN`
* `Simple_CNN`
* `Complex_CNN`
* `RNN_lstm`
* `Novel_CNN`

同时也包含传统 baseline 方法：

* Butterworth / notch filter-based denoising
* EMD-based denoising

---

## Project Structure

```text
EEGdenoiseNet/
├── baselines/
│   ├── __init__.py
│   ├── emd_baseline.py
│   └── filter_baseline.py
├── data/
│   ├── EEG_all_epochs.mat
│   ├── EEG_all_epochs.npy
│   ├── EMG_all_epochs.mat
│   ├── EMG_all_epochs.npy
│   ├── EOG_all_epochs.mat
│   └── EOG_all_epochs.npy
├── models/
│   ├── __init__.py
│   ├── complex_cnn.py
│   ├── fc_nn.py
│   ├── novel_cnn.py
│   ├── rnn_lstm.py
│   └── simple_cnn.py
├── preprocessing/
│   ├── __init__.py
│   ├── augment.py
│   └── preparation.py
├── results/
├── scripts/
│   └── run_train.py
├── training/
│   ├── __init__.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── save_method.py
│   └── trainer.py
├── visualization/
│   ├── __init__.py
│   ├── plot_examples.py
│   ├── plot_loss.py
│   └── plot_metrics.py
├── main.py
├── pipeline.py
├── environment.yml
├── LICENSE
└── README.md
```

说明：

* `data/`：存放 EEG、EOG、EMG 数据文件。
* `models/`：存放深度学习模型结构。
* `preprocessing/`：负责数据划分、噪声混合、标准化和数据增强。
* `training/`：负责训练、测试、损失函数、优化器和结果保存。
* `baselines/`：传统去噪 baseline 方法。
* `visualization/`：绘制波形、PSD、loss 曲线和指标图。
* `results/`：保存训练结果、去噪结果和可视化结果。
* `scripts/run_train.py`：推荐使用的命令行运行入口。
* `main.py`：保留的旧版入口。
* `pipeline.py`：整合数据加载、baseline、深度模型训练和结果保存的主流程。

---

## Environment Setup

推荐使用 Conda 创建虚拟环境。

项目要求 Python 版本为：

```text
Python 3.10
```

使用下面的命令创建环境：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate eegdenoise
```

如果需要手动安装依赖，可以参考以下主要依赖：

```text
python=3.10
numpy
scipy
matplotlib
pytorch
EMD-signal
```

注意：

`EMD-signal` 是安装包名称，但代码中导入方式为：

```python
from PyEMD import EMD
```

不要安装成 `pyemd`，那是另一个不同的包。

---

## Data Preparation

请将数据文件放入 `data/` 目录下。

默认需要以下 `.npy` 文件：

```text
data/
├── EEG_all_epochs.npy
├── EOG_all_epochs.npy
└── EMG_all_epochs.npy
```

项目中也可以保留对应的 `.mat` 文件：

```text
data/
├── EEG_all_epochs.mat
├── EOG_all_epochs.mat
└── EMG_all_epochs.mat
```

默认运行时会从 `data/` 目录读取数据。

---

## Usage

推荐使用命令行入口：

```bash
python scripts/run_train.py
```

如果未指定噪声类型，程序会进入交互模式，让用户选择：

```text
1. EOG
2. EMG
3. EOG_EMG
```

---

## Run Full Pipeline

运行完整流程，包括传统 baseline 和深度学习模型：

```bash
python scripts/run_train.py --noise-type EOG
```

指定数据目录和结果保存目录：

```bash
python scripts/run_train.py \
    --noise-type EOG \
    --data-dir ./data \
    --result-dir ./results
```

运行混合噪声场景：

```bash
python scripts/run_train.py --noise-type EOG_EMG
```

---

## Train Specific Deep Models

只训练指定模型，例如 `Novel_CNN`：

```bash
python scripts/run_train.py \
    --noise-type EMG \
    --models Novel_CNN
```

同时训练多个模型：

```bash
python scripts/run_train.py \
    --noise-type EOG \
    --models Simple_CNN Complex_CNN Novel_CNN
```

如果不指定 `--models`，默认会运行所有深度学习模型：

```text
fcNN
Simple_CNN
Complex_CNN
RNN_lstm
Novel_CNN
```

---

## Run Traditional Baselines Only

只运行传统去噪方法：

```bash
python scripts/run_train.py \
    --noise-type EOG \
    --traditional-only
```

传统方法包括：

* filter-based denoising
* EMD-based denoising

---

## Run Deep Models Only

只运行深度学习模型：

```bash
python scripts/run_train.py \
    --noise-type EMG \
    --deep-only
```

指定模型并只运行深度学习部分：

```bash
python scripts/run_train.py \
    --noise-type EOG_EMG \
    --deep-only \
    --models Novel_CNN
```

---

## Common Arguments

| Argument             | Description                   | Default     |
| -------------------- | ----------------------------- | ----------- |
| `--data-dir`         | 数据文件夹路径                       | `./data`    |
| `--result-dir`       | 结果保存路径                        | `./results` |
| `--noise-type`       | 噪声类型，可选 `EOG`、`EMG`、`EOG_EMG` | 交互选择        |
| `--models`           | 指定要运行的深度学习模型                  | 全部模型        |
| `--epochs`           | 训练轮数                          | `30`        |
| `--batch-size`       | batch size                    | `25`        |
| `--combin-num`       | 每个 EEG 片段的随机噪声组合次数            | `10`        |
| `--train-per`        | 训练集比例                         | `0.8`       |
| `--optimizer`        | 优化器，可选 `Adam`、`RMSprop`、`SGD` | `Adam`      |
| `--traditional-only` | 只运行传统 baseline                | `False`     |
| `--deep-only`        | 只运行深度学习模型                     | `False`     |
| `--train-num`        | 实验编号，用于结果子文件夹命名               | `1`         |

示例：

```bash
python scripts/run_train.py \
    --noise-type EOG \
    --epochs 50 \
    --batch-size 32 \
    --optimizer Adam \
    --train-num 1
```

---

## Legacy Entry Point

项目也保留了 `main.py` 作为旧版入口：

```bash
python main.py
```

不过更推荐使用：

```bash
python scripts/run_train.py
```

因为 `scripts/run_train.py` 支持更多命令行参数，方便进行不同实验配置。

---

## Output Results

实验结果默认保存到：

```text
results/
```

根据噪声类型、模型名称和实验编号，程序会生成对应的结果子目录。

结果内容通常包括：

* 去噪后的 EEG 信号
* loss 曲线
* 不同 SNR 下的评价指标
* 示例波形图
* PSD 对比图
* baseline 和 deep learning model 的结果对比

---

## Visualization

`visualization/` 文件夹中包含绘图函数：

```text
visualization/
├── plot_examples.py
├── plot_loss.py
└── plot_metrics.py
```

主要功能包括：

* 绘制原始 EEG、含噪 EEG、去噪 EEG 的波形对比
* 绘制功率谱密度 PSD 对比
* 绘制训练 loss 曲线
* 绘制不同 SNR 下的评价指标曲线

---

## Import Notes

本项目以 `EEGdenoiseNet/` 作为项目根目录运行。

推荐从项目根目录执行命令：

```bash
python main.py
```

或：

```bash
python scripts/run_train.py
```



## Troubleshooting

### 1. Cannot import `PyEMD`

如果出现：

```text
ModuleNotFoundError: No module named 'PyEMD'
```

请安装：

```bash
pip install EMD-signal
```

或者重新创建 Conda 环境：

```bash
conda env create -f environment.yml
```

---

### 2. CUDA / PyTorch 问题

如果需要使用 GPU，请根据服务器 CUDA 版本安装对应的 PyTorch 版本。

可以先检查 PyTorch 是否可用：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

---

## License

This project is released under the license specified in the `LICENSE` file.

---

## Suggested Citation

If this project is used for academic work, please cite the original EEGdenoiseNet-related literature and clearly describe any modifications made in this implementation.

---

## Acknowledgements

This project is inspired by EEG denoising research using both traditional signal processing methods and deep learning methods. It includes experimental comparisons between conventional baselines and neural network-based denoising models.
