# NanoGPT

一个从零开始学习 GPT 架构的实验项目。

当前实现了一个字符级、decoder-only 的 GPT 风格语言模型：从文本编码、训练集与验证集划分、批次采样开始，逐步加入 token 与 position embedding、因果自注意力、多头注意力、前馈网络、LayerNorm、Dropout、残差连接、AdamW 训练和自回归文本生成。

项目参考 Andrej Karpathy 的 nanoGPT，并以理解 Transformer 各个组件的实现和作用为目标，后续会继续向更完整的 GPT-2 结构扩展。

## Reference

This learning repository follows the ideas and implementation progression in Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repository. The code here is written and extended as a learning exercise, beginning with a character-level language model and gradually building a GPT-style Transformer.

## 项目结构

```text
.
├── NanoGPT.py       # 独立训练脚本
├── train.ipynb      # 分步骤实验 notebook
└── data/input.txt   # 训练文本
```

## 环境

需要 Python 以及 PyTorch。GPU 训练需要安装与 NVIDIA 驱动兼容的 CUDA 版 PyTorch。

```bash
python -m pip install torch
```

安装完成后可以检查 PyTorch 是否识别 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 运行

在仓库根目录执行：

```bash
python NanoGPT.py
```

也可以打开 `train.ipynb`，按 notebook 中的步骤观察数据处理、模型计算和训练过程。

## 学习内容

- 字符表、encoder 和 decoder
- token embedding 与 position embedding
- logits、softmax 和交叉熵
- 训练集与验证集
- 反向传播、AdamW 和文本生成

这是一个进行中的学习项目，模型规模、训练配置和实现细节会随着学习过程继续调整。
