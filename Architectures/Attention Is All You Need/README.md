# Transformer Implementation from Scratch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Research Paper](https://img.shields.io/badge/Research_Paper-%231E90FF.svg?logo=Google-Scholar&logoColor=white)](https://arxiv.org/abs/1706.03762)

A PyTorch implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al.), featuring multi-head attention, positional embeddings, and encoder-decoder structure.

## Architecture Overview

```text
Transformer
├── Encoder
│   ├── Input Embedding (Word + Positional)
│   ├── N× Encoder Layers
│   │   ├── Multi-Head Self-Attention
│   │   ├── LayerNorm
│   │   ├── Position-wise FFN
│   │   └── Residual Connections
│   └── Output (Context Vectors)
│
└── Decoder
    ├── Input Embedding (Word + Positional)
    ├── N× Decoder Layers
    │   ├── Masked Multi-Head Self-Attention
    │   ├── LayerNorm
    │   ├── Encoder-Decoder Attention
    │   ├── Position-wise FFN
    │   └── Residual Connections
    └── Output Projection (Linear + Softmax)
```

## Configuration Options

| Parameter       | Description                          | Default Value |
|-----------------|--------------------------------------|--------------:|
| `d_model`       | Dimension of the model               | 512           |
| `nx`            | Number of encoder/decoder layers     | 6             |
| `heads`         | Number of attention heads           | 8             |
| `d_ff`          | Inner dimension of feed-forward network | 2048        |
| `dropout`       | Dropout probability                  | 0.1           |
| `max_length`    | Maximum sequence length              | 100           |

