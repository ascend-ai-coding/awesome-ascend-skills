#!/usr/bin/env python3
"""
逐层输出调试脚本 - NPU/PyTorch 版
在 NPU 机器上运行，打印每层的 mean/std 便于与 GPU 对比。
"""
import os, sys, numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs

PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')

model, n_ann = convert_tf_to_pytorch(PKL_PATH)
model.eval()

seqs = [
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE',
    'ACDEFGHIKLMNPQRSTUVWXY',
]
seq_len = 512
tokens = tokenize_seqs(seqs, seq_len)
inp_seq = torch.from_numpy(tokens).long()
inp_ann = torch.zeros(3, n_ann)

def p(name, t):
    a = t.detach().numpy() if isinstance(t, torch.Tensor) else t
    print(f'{name:45s} shape={str(a.shape):25s} mean={a.mean():.6f} std={a.std():.6f} min={a.min():.6f} max={a.max():.6f}')

with torch.no_grad():
    hidden_seq = model.seq_embedding(inp_seq)
    p('embedding-seq-input', hidden_seq)

    hidden_global = F.gelu(model.global_input_dense(inp_ann))
    p('dense-global-input', hidden_global)

    for i, block in enumerate(model.blocks):
        bi = i + 1
        seqed_global = F.gelu(block.global_to_seq_dense(hidden_global)).unsqueeze(1)
        seq_t = hidden_seq.transpose(1, 2)
        narrow = F.gelu(block.narrow_conv(seq_t)).transpose(1, 2)
        wide = F.gelu(block.wide_conv(seq_t)).transpose(1, 2)

        hidden_seq = block.seq_norm1(hidden_seq + seqed_global + narrow + wide)
        p(f'seq-merge1-norm-block{bi}', hidden_seq)

        hidden_seq = block.seq_norm2(hidden_seq + F.gelu(block.seq_dense(hidden_seq)))
        p(f'seq-merge2-norm-block{bi}', hidden_seq)

        dense_g = F.gelu(block.global_dense1(hidden_global))
        attn = block.global_attention(hidden_global, hidden_seq)
        hidden_global = block.global_norm1(hidden_global + dense_g + attn)
        p(f'global-merge1-norm-block{bi}', hidden_global)

        hidden_global = block.global_norm2(hidden_global + F.gelu(block.global_dense2(hidden_global)))
        p(f'global-merge2-norm-block{bi}', hidden_global)

    out_seq = F.softmax(model.output_seq_dense(hidden_seq), dim=-1)
    p('output-seq', out_seq)

    out_ann = torch.sigmoid(model.output_annotations_dense(hidden_global))
    p('output-annotations', out_ann)
