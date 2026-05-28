# Results 文件夹说明

本文件夹用于保存 EEGdenoiseNet 训练、测试和可视化分析过程中生成的结果文件。每一次模型训练或传统方法测试都会在 `results/` 下生成一个独立的结果目录，便于后续复现实验、比较不同模型、分析去噪效果。

> 建议：实际实验输出通常较大，不建议直接上传到 GitHub。仓库中可以只保留本文件 `results/README.md`，用于说明结果目录结构和分析方法。

---

## 1. 结果目录整体结构

典型结构如下：

```text
results/
├── EMG_Complex_CNN_Adam/
│   └── 1/
│       ├── denoise_model/
│       │   ├── model_full.pt
│       │   └── model_state_dict.pt
│       ├── figures/
│       │   ├── example_waveform_psd.png
│       │   ├── loss_curve.png
│       │   └── metrics_by_snr.png
│       ├── nn_output/
│       │   ├── Denoiseoutput_test.npy
│       │   ├── EEG_test.npy
│       │   ├── loss_history.npy
│       │   ├── metrics_by_snr.npy
│       │   ├── noiseinput_test.npy
│       │   └── test_std_VALUE.npy
│       ├── test/
│       └── train/
├── EMG_fcNN_Adam/
│   └── 1/
│       ├── denoise_model/
│       ├── figures/
│       └── nn_output/
└── ...
```

目录命名通常遵循以下格式：

```text
<NoiseType>_<ModelName>_<Optimizer>/<RunID>/
```

例如：

```text
EMG_Complex_CNN_Adam/1/
```

表示：

```text
噪声类型：EMG
模型名称：Complex_CNN
优化器：Adam
实验编号：1
```

常见噪声类型包括：

```text
EOG
EMG
EOG_EMG
```

常见深度学习模型包括：

```text
fcNN
Simple_CNN
Complex_CNN
RNN_lstm
Novel_CNN
```

传统方法结果可能会以类似方式命名：

```text
EOG_Traditional/
EMG_Traditional/
EOG_EMG_Traditional/
```

---

## 2. 各子文件夹说明

### 2.1 `denoise_model/`

该文件夹用于保存训练后的模型文件。

```text
denoise_model/
├── model_full.pt
└── model_state_dict.pt
```

#### `model_full.pt`

保存完整模型对象，包括模型结构和参数。

优点是加载方便：

```python
import torch

model = torch.load("model_full.pt")
model.eval()
```

缺点是对代码结构依赖较强。如果模型类路径发生变化，可能无法正常加载。

#### `model_state_dict.pt`

只保存模型参数，不保存模型结构。

推荐用于长期保存和复现实验：

```python
import torch
from models import build_model

model = build_model("Complex_CNN", datanum=1024)
model.load_state_dict(torch.load("model_state_dict.pt"))
model.eval()
```

这种方式更稳定，也更适合用于论文复现实验。

---

### 2.2 `figures/`

该文件夹保存自动生成的可视化结果。

```text
figures/
├── example_waveform_psd.png
├── loss_curve.png
└── metrics_by_snr.png
```

#### `loss_curve.png`

训练损失曲线图，主要用于分析模型训练过程。

可以观察：

- loss 是否稳定下降
- 是否出现震荡
- 是否存在过拟合
- 训练轮数是否足够
- 学习率是否合适

一般判断方式：

```text
训练 loss 持续下降：模型在学习
验证 loss 同步下降：训练过程较正常
训练 loss 下降但验证 loss 上升：可能过拟合
loss 长期不下降：模型、数据或学习率可能存在问题
```

#### `example_waveform_psd.png`

示例波形和功率谱密度图，用于定性分析去噪效果。

图中通常会对比：

```text
Noisy EEG
Clean EEG
Denoised EEG
```

可以从两个角度分析：

1. **时域波形**  
   观察去噪信号是否更接近干净 EEG，是否保留主要波形结构，是否出现过平滑或异常尖峰。

2. **频域 PSD**  
   观察去噪后信号的频谱是否更接近干净 EEG，是否有效抑制 EOG 或 EMG 噪声成分。

#### `metrics_by_snr.png`

不同 SNR 条件下的模型指标图，主要用于分析模型在不同噪声强度下的鲁棒性。

重点关注：

```text
低 SNR 时模型是否仍然稳定
高 SNR 时模型是否接近 clean EEG
不同模型在低噪声和高噪声条件下的性能差异
```

如果某个模型平均指标很好，但低 SNR 下性能下降明显，说明它的鲁棒性可能不足。

---

### 2.3 `nn_output/`

该文件夹保存模型测试阶段的核心数值结果，是后续分析最重要的部分。

```text
nn_output/
├── Denoiseoutput_test.npy
├── EEG_test.npy
├── loss_history.npy
├── metrics_by_snr.npy
├── noiseinput_test.npy
└── test_std_VALUE.npy
```

#### `noiseinput_test.npy`

测试集中的含噪 EEG 输入信号，可以理解为模型输入：

```text
Noisy EEG
```

#### `EEG_test.npy`

测试集中的干净 EEG 信号，可以理解为 ground truth：

```text
Clean EEG
```

这是评价模型效果时的参考标准。

#### `Denoiseoutput_test.npy`

模型输出的去噪 EEG 信号，可以理解为：

```text
Denoised EEG
```

后续所有评价指标基本都围绕下面三者展开：

```text
noiseinput_test.npy      含噪输入
EEG_test.npy             干净参考信号
Denoiseoutput_test.npy   模型去噪输出
```

核心问题是：

```text
Denoiseoutput_test 是否比 noiseinput_test 更接近 EEG_test
```

#### `loss_history.npy`

训练过程中的 loss 记录。

可用于重新绘制 loss 曲线，或比较不同模型的收敛速度。

可以分析：

```text
最终 loss
最低 loss
收敛速度
是否震荡
是否过拟合
```

#### `metrics_by_snr.npy`

不同 SNR 条件下的评价指标结果。

通常用于绘制：

```text
metrics_by_snr.png
```

也可以用于生成论文或报告中的表格。

常见分析指标包括：

```text
RRMSE
RMSE
CC
SNR improvement
```

一般判断标准：

```text
RRMSE / RMSE 越低越好
CC 越高越好
SNR improvement 越高越好
```

#### `test_std_VALUE.npy`

测试集标准化或反标准化所需的标准差相关数值。

该文件通常用于把模型输出从标准化尺度恢复到原始信号尺度。在进行可视化、指标计算或保存最终去噪信号时，应注意是否需要使用该文件进行反标准化。

---

### 2.4 `train/` 和 `test/`

这两个文件夹通常用于保存训练集和测试集相关的中间文件或后续扩展结果。

如果当前为空，可以暂时保留。后续可以考虑用于保存：

```text
训练集输入
训练集标签
测试集输入
测试集标签
测试集预测结果
分样本误差
分 SNR 结果
```

---

## 3. 推荐开展的分析方法

基于当前结果文件，可以开展以下几类分析。

---

### 3.1 训练过程分析

主要使用：

```text
loss_history.npy
loss_curve.png
```

分析目标：

```text
判断模型是否正常收敛
判断训练轮数是否足够
判断是否存在过拟合
比较不同模型的收敛速度
```

建议比较内容：

```text
fcNN vs Simple_CNN
Simple_CNN vs Complex_CNN
Complex_CNN vs Novel_CNN
CNN 模型 vs RNN_lstm
```

可回答的问题：

```text
哪个模型收敛最快？
哪个模型训练最稳定？
哪个模型最终 loss 最低？
是否存在明显过拟合？
```

---

### 3.2 整体去噪效果分析

主要使用：

```text
noiseinput_test.npy
EEG_test.npy
Denoiseoutput_test.npy
```

建议计算整体指标：

```text
RMSE
RRMSE
相关系数 CC
SNR improvement
```

可以整理成表格：

```text
Noise Type | Model       | RMSE ↓ | RRMSE ↓ | CC ↑ | SNR Improvement ↑
EMG        | fcNN        |        |         |      |
EMG        | Simple_CNN  |        |         |      |
EMG        | Complex_CNN |        |         |      |
EMG        | RNN_lstm    |        |         |      |
EMG        | Novel_CNN   |        |         |      |
```

分析重点：

```text
哪个模型整体去噪效果最好
深度学习方法是否优于传统滤波和 EMD
CNN 类模型是否优于全连接网络
复杂模型是否真的带来更好性能
```

---

### 3.3 按 SNR 分组的鲁棒性分析

主要使用：

```text
metrics_by_snr.npy
metrics_by_snr.png
```

分析目标：

```text
评估模型在不同噪声强度下的稳定性
```

重点观察：

```text
低 SNR 条件下模型是否仍然有效
高 SNR 条件下不同模型差距是否缩小
哪个模型对强噪声最鲁棒
哪个模型性能随 SNR 变化最平稳
```

建议重点分析低 SNR 区间，因为低 SNR 场景更能体现模型能力。

例如：

```text
在低 SNR 条件下，如果 Complex_CNN 的 RRMSE 明显低于 fcNN，
说明卷积结构能够更好地提取局部时序特征并抑制噪声。
```

---

### 3.4 时域波形分析

主要使用：

```text
example_waveform_psd.png
noiseinput_test.npy
EEG_test.npy
Denoiseoutput_test.npy
```

建议选取若干代表性样本：

```text
低 SNR 样本
中等 SNR 样本
高 SNR 样本
```

每个样本画三条曲线：

```text
Noisy EEG
Clean EEG
Denoised EEG
```

分析重点：

```text
Denoised EEG 是否更接近 Clean EEG
是否有效去除了大幅度噪声
是否保留 EEG 原始波形结构
是否出现过平滑
是否引入新的异常波动
```

这类图适合用于论文、报告或答辩展示。

---

### 3.5 频域 PSD 分析

主要使用：

```text
example_waveform_psd.png
```

也可以基于 `.npy` 文件重新计算 PSD。

分析重点：

```text
去噪后频谱是否接近 clean EEG
是否抑制了噪声主导频段
是否错误削弱了 EEG 有效频段
```

不同噪声类型可以重点关注不同频段：

```text
EOG 噪声：通常更偏低频
EMG 噪声：通常更偏高频
EOG_EMG：同时包含低频和高频干扰
```

如果一个模型时域波形看起来很好，但 PSD 明显偏离 clean EEG，说明它可能存在频域失真。

---

### 3.6 不同噪声类型对比分析

如果分别运行了：

```text
EOG
EMG
EOG_EMG
```

可以比较同一模型在不同噪声下的表现。

建议整理表格：

```text
Model       | EOG RRMSE ↓ | EMG RRMSE ↓ | EOG_EMG RRMSE ↓
fcNN        |             |             |
Simple_CNN  |             |             |
Complex_CNN |             |             |
RNN_lstm    |             |             |
Novel_CNN   |             |             |
```

可回答的问题：

```text
哪一种噪声最难去除？
模型对 EOG 和 EMG 的适应性是否不同？
混合噪声 EOG_EMG 是否显著降低模型性能？
```

一般来说，混合噪声任务更复杂，应重点关注模型在 `EOG_EMG` 下的鲁棒性。

---

### 3.7 不同模型横向比较

可以基于各模型文件夹中的结果进行横向比较。

例如：

```text
EMG_fcNN_Adam/1/
EMG_Simple_CNN_Adam/1/
EMG_Complex_CNN_Adam/1/
EMG_RNN_lstm_Adam/1/
EMG_Novel_CNN_Adam/1/
```

建议比较：

```text
整体指标
低 SNR 指标
loss 收敛速度
模型输出波形
PSD 保真度
模型参数量和训练耗时
```

最终可以回答：

```text
哪个模型效果最好？
哪个模型最稳定？
哪个模型计算成本最高？
哪个模型最适合实际应用？
```

---

## 4. 推荐的最终结果整理方式

为了方便写论文、报告或 GitHub README，建议最终从 `results/` 中整理出以下几类材料。

### 4.1 总体指标表

推荐保存为：

```text
summary_metrics.csv
```

内容示例：

```text
Noise Type,Model,Optimizer,Run ID,RMSE,RRMSE,CC,SNR Improvement
EMG,fcNN,Adam,1,...
EMG,Complex_CNN,Adam,1,...
```

这是最重要的结果汇总文件。

---

### 4.2 按 SNR 分组指标表

推荐保存为：

```text
summary_metrics_by_snr.csv
```

内容示例：

```text
Noise Type,Model,SNR,RMSE,RRMSE,CC,SNR Improvement
EMG,fcNN,-7,...
EMG,fcNN,-6,...
EMG,Complex_CNN,-7,...
EMG,Complex_CNN,-6,...
```

该文件适合用于绘制模型鲁棒性图。

---

### 4.3 代表性波形图

建议从每个噪声类型中选取若干样本：

```text
low_snr_example.png
medium_snr_example.png
high_snr_example.png
```

用于展示不同噪声强度下的去噪效果。

---

### 4.4 频谱分析图

建议保存：

```text
psd_comparison.png
```

用于展示 noisy、clean、denoised 三者在频域上的差异。

---

## 5. 文件保留建议

建议长期保留：

```text
model_state_dict.pt
Denoiseoutput_test.npy
EEG_test.npy
noiseinput_test.npy
loss_history.npy
metrics_by_snr.npy
loss_curve.png
metrics_by_snr.png
example_waveform_psd.png
```

可以选择性保留：

```text
model_full.pt
```

如果仓库空间有限，不建议上传：

```text
*.pt
*.npy
results/
```

这些文件通常较大，建议本地保存或通过网盘、实验记录系统单独管理。

GitHub 仓库中建议只保留：

```text
results/README.md
```

而不上传实际实验输出。

---

## 6. 建议的分析顺序

推荐按以下顺序进行结果分析：

```text
1. 检查每个模型的 loss_curve.png，确认训练是否正常
2. 查看 metrics_by_snr.png，比较不同 SNR 下的鲁棒性
3. 使用 metrics_by_snr.npy 生成模型对比表
4. 使用 noiseinput_test.npy、EEG_test.npy、Denoiseoutput_test.npy 重新计算整体指标
5. 查看 example_waveform_psd.png，进行时域和频域定性分析
6. 横向比较不同模型、不同噪声类型、不同优化器
7. 汇总出最终的 summary_metrics.csv 和 summary_metrics_by_snr.csv
```

---

## 7. 简要结论模板

在撰写论文或报告时，可以按照以下模板描述实验结果：

```text
首先，通过 loss 曲线观察各模型的训练收敛情况。随后，基于测试集中的 noisy EEG、clean EEG 和 denoised EEG 计算 RMSE、RRMSE、CC 以及 SNR improvement 等指标。进一步地，按照不同输入 SNR 分组统计模型性能，以评估模型在不同噪声强度下的鲁棒性。最后，通过典型样本的时域波形和 PSD 频谱对比，对模型去噪效果进行定性分析。

综合整体指标、低 SNR 条件下的表现、时域波形保真度和频域特征保持情况，可以评估不同模型在 EEG 去噪任务中的有效性和适用场景。
```