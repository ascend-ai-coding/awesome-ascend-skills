#!/usr/bin/env python3
"""
ProteinBERT 权重转换工具（独立脚本）

将原始 TF pickle 权重转换为 PyTorch state_dict，可在任意机器上运行。
仅依赖 torch + numpy，无需 TensorFlow。

用法:
    python convert_weights.py \
        --input ~/proteinbert_models/epoch_92400_sample_23500000.pkl \
        --output ~/proteinbert_models/proteinbert_pytorch.pt
"""
import argparse
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PyTorch 版 ProteinBERT 模型定义（自包含）
# ============================================================

class GlobalAttention(nn.Module):
    def __init__(self, n_heads, d_key, d_value, d_global_input, d_seq_input):
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.d_output = n_heads * d_value
        self.sqrt_d_key = math.sqrt(d_key)
        self.Wq = nn.Parameter(torch.empty(n_heads, d_global_input, d_key))
        self.Wk = nn.Parameter(torch.empty(n_heads, d_seq_input, d_key))
        self.Wv = nn.Parameter(torch.empty(n_heads, d_seq_input, d_value))
        nn.init.xavier_uniform_(self.Wq.view(-1, d_key))
        nn.init.xavier_uniform_(self.Wk.view(-1, d_key))
        nn.init.xavier_uniform_(self.Wv.view(-1, d_value))

    def forward(self, global_repr, seq_repr):
        batch_size = global_repr.size(0)
        length = seq_repr.size(1)
        VS = F.gelu(torch.einsum('bls,hsv->bhlv', seq_repr, self.Wv))
        VS = VS.reshape(batch_size * self.n_heads, length, self.d_value)
        QX = torch.tanh(torch.einsum('bg,hgk->bhk', global_repr, self.Wq))
        QX = QX.reshape(batch_size * self.n_heads, self.d_key)
        KS = torch.tanh(torch.einsum('bls,hsk->bhkl', seq_repr, self.Wk))
        KS = KS.reshape(batch_size * self.n_heads, self.d_key, length)
        attn = F.softmax(torch.bmm(QX.unsqueeze(1), KS).squeeze(1) / self.sqrt_d_key, dim=-1)
        Y = torch.bmm(attn.unsqueeze(1), VS).squeeze(1)
        return Y.reshape(batch_size, self.d_output)


class ProteinBERTBlock(nn.Module):
    def __init__(self, d_seq, d_global, n_heads, d_key, kernel, dilation):
        super().__init__()
        d_value = d_global // n_heads
        self.global_to_seq_dense = nn.Linear(d_global, d_seq)
        self.narrow_conv = nn.Conv1d(d_seq, d_seq, kernel_size=kernel, padding='same', dilation=1)
        self.wide_conv = nn.Conv1d(d_seq, d_seq, kernel_size=kernel, padding='same', dilation=dilation)
        self.seq_norm1 = nn.LayerNorm(d_seq, eps=1e-3)
        self.seq_dense = nn.Linear(d_seq, d_seq)
        self.seq_norm2 = nn.LayerNorm(d_seq, eps=1e-3)
        self.global_dense1 = nn.Linear(d_global, d_global)
        self.global_attention = GlobalAttention(n_heads, d_key, d_value, d_global, d_seq)
        self.global_norm1 = nn.LayerNorm(d_global, eps=1e-3)
        self.global_dense2 = nn.Linear(d_global, d_global)
        self.global_norm2 = nn.LayerNorm(d_global, eps=1e-3)

    def forward(self, hidden_seq, hidden_global):
        seqed_global = F.gelu(self.global_to_seq_dense(hidden_global)).unsqueeze(1)
        seq_t = hidden_seq.transpose(1, 2)
        narrow = F.gelu(self.narrow_conv(seq_t)).transpose(1, 2)
        wide = F.gelu(self.wide_conv(seq_t)).transpose(1, 2)
        hidden_seq = self.seq_norm1(hidden_seq + seqed_global + narrow + wide)
        hidden_seq = self.seq_norm2(hidden_seq + F.gelu(self.seq_dense(hidden_seq)))
        dense_g = F.gelu(self.global_dense1(hidden_global))
        attn = self.global_attention(hidden_global, hidden_seq)
        hidden_global = self.global_norm1(hidden_global + dense_g + attn)
        hidden_global = self.global_norm2(hidden_global + F.gelu(self.global_dense2(hidden_global)))
        return hidden_seq, hidden_global


class ProteinBERTModel(nn.Module):
    def __init__(self, vocab_size=26, n_annotations=8943, d_hidden_seq=128,
                 d_hidden_global=512, n_blocks=6, n_heads=4, d_key=64,
                 conv_kernel_size=9, wide_conv_dilation_rate=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_annotations = n_annotations
        self.d_hidden_seq = d_hidden_seq
        self.d_hidden_global = d_hidden_global
        self.n_blocks = n_blocks
        self.seq_embedding = nn.Embedding(vocab_size, d_hidden_seq)
        self.global_input_dense = nn.Linear(n_annotations, d_hidden_global)
        self.blocks = nn.ModuleList([
            ProteinBERTBlock(d_hidden_seq, d_hidden_global, n_heads, d_key,
                             conv_kernel_size, wide_conv_dilation_rate)
            for _ in range(n_blocks)
        ])
        self.output_seq_dense = nn.Linear(d_hidden_seq, vocab_size)
        self.output_annotations_dense = nn.Linear(d_hidden_global, n_annotations)

    def forward(self, input_seq, input_annotations, return_hidden=False):
        hidden_seq = self.seq_embedding(input_seq)
        hidden_global = F.gelu(self.global_input_dense(input_annotations))
        seq_hiddens, global_hiddens = [], []
        for block in self.blocks:
            hidden_seq, hidden_global = block(hidden_seq, hidden_global)
            if return_hidden:
                seq_hiddens.append(hidden_seq)
                global_hiddens.append(hidden_global)
        output_seq = F.softmax(self.output_seq_dense(hidden_seq), dim=-1)
        output_annotations = torch.sigmoid(self.output_annotations_dense(hidden_global))
        if return_hidden:
            return output_seq, output_annotations, seq_hiddens, global_hiddens
        return output_seq, output_annotations


# ============================================================
# 权重转换逻辑
# ============================================================

def convert(pkl_path, output_path):
    print(f'[1/3] 读取 TF 权重: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        n_annotations, model_weights, _ = pickle.load(f)
    print(f'      n_annotations={n_annotations}, 权重数组数={len(model_weights)}')

    print('[2/3] 构建 PyTorch 模型并映射权重...')
    model = ProteinBERTModel(vocab_size=26, n_annotations=n_annotations)
    state_dict = model.state_dict()
    weight_map = {}
    idx = 0

    def take(n=1):
        nonlocal idx
        result = model_weights[idx:idx + n]
        idx += n
        return result[0] if n == 1 else result

    weight_map['global_input_dense.weight'] = take().T
    weight_map['global_input_dense.bias'] = take()
    weight_map['seq_embedding.weight'] = take()

    for bi in range(model.n_blocks):
        p = f'blocks.{bi}'
        weight_map[f'{p}.global_to_seq_dense.weight'] = take().T
        weight_map[f'{p}.global_to_seq_dense.bias'] = take()
        weight_map[f'{p}.narrow_conv.weight'] = np.transpose(take(), (2, 1, 0))
        weight_map[f'{p}.narrow_conv.bias'] = take()
        weight_map[f'{p}.wide_conv.weight'] = np.transpose(take(), (2, 1, 0))
        weight_map[f'{p}.wide_conv.bias'] = take()
        weight_map[f'{p}.seq_norm1.weight'] = take()
        weight_map[f'{p}.seq_norm1.bias'] = take()
        weight_map[f'{p}.seq_dense.weight'] = take().T
        weight_map[f'{p}.seq_dense.bias'] = take()
        weight_map[f'{p}.seq_norm2.weight'] = take()
        weight_map[f'{p}.seq_norm2.bias'] = take()
        weight_map[f'{p}.global_dense1.weight'] = take().T
        weight_map[f'{p}.global_dense1.bias'] = take()
        weight_map[f'{p}.global_attention.Wq'] = take()
        weight_map[f'{p}.global_attention.Wk'] = take()
        weight_map[f'{p}.global_attention.Wv'] = take()
        weight_map[f'{p}.global_norm1.weight'] = take()
        weight_map[f'{p}.global_norm1.bias'] = take()
        weight_map[f'{p}.global_dense2.weight'] = take().T
        weight_map[f'{p}.global_dense2.bias'] = take()
        weight_map[f'{p}.global_norm2.weight'] = take()
        weight_map[f'{p}.global_norm2.bias'] = take()

    weight_map['output_seq_dense.weight'] = take().T
    weight_map['output_seq_dense.bias'] = take()
    weight_map['output_annotations_dense.weight'] = take().T
    weight_map['output_annotations_dense.bias'] = take()

    print(f'      已映射 {idx}/{len(model_weights)} 个 TF 权重 → {len(weight_map)} 个 PyTorch 参数')

    for key, val in weight_map.items():
        tensor = torch.from_numpy(val)
        assert state_dict[key].shape == tensor.shape, \
            f'Shape mismatch: {key} expected {state_dict[key].shape}, got {tensor.shape}'
        state_dict[key] = tensor

    model.load_state_dict(state_dict)

    print(f'[3/3] 保存 PyTorch 模型: {output_path}')
    torch.save({
        'n_annotations': n_annotations,
        'vocab_size': 26,
        'state_dict': model.state_dict(),
    }, output_path)
    print(f'      完成！文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')
    return model, n_annotations


# ============================================================
# 入口
# ============================================================

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProteinBERT TF→PyTorch 权重转换')
    parser.add_argument('--input', '-i', required=True,
                        help='TF 原始权重 pkl 文件路径')
    parser.add_argument('--output', '-o', required=True,
                        help='PyTorch 权重输出路径 (.pt)')
    args = parser.parse_args()

    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    convert(args.input, args.output)
