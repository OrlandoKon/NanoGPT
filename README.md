# NanoGPT

一个从零开始学习 GPT 架构的实验项目。

当前版本从字符级语言模型开始，实现了文本编码、训练集与验证集划分、批次采样、Bigram language model、交叉熵损失、AdamW 优化和文本生成。项目中的代码会逐步扩展到更完整的 GPT-2 结构，重点是理解每个组件的作用，而不是直接使用现成的模型实现。

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
