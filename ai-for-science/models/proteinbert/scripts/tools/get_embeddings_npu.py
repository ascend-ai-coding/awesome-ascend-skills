#!/usr/bin/env python3
"""
ProteinBERT Embedding 提取 - PyTorch NPU 版

对标 GitHub README 中的原始 TF 用法。
精确复刻 TF 版 get_model_with_hidden_layers_as_outputs 的层收集逻辑。

TF 版实际收集的层（通过源码分析确认）:
  seq 方向: 仅 LayerNorm(3D) + output-seq，共 12+1=13 层 -> 12*128 + 26 = 1562 维
    注意: embedding 层名为 'embedding-seq-input'，不在过滤名单中，所以不收集
  global 方向: dense-global-input + LayerNorm(2D) + output-annotations，共 1+12+1=14 层 -> 512 + 12*512 + 8943 = 15599 维
    注意: input_annotations (下划线) 与实际 Input 层名 'input-annotations' (连字符) 不匹配，所以不收集

用法:
    source /home/Ascend/ascend-toolkit/set_env.sh
    python get_embeddings_npu.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs

PKL_PATH = os.path.expanduser("~/proteinbert_models/epoch_92400_sample_23500000.pkl")
DEVICE = "npu:0"


def get_all_hidden_representations(model, input_seq, input_annotations):
    """
    精确对齐 TF 版 get_model_with_hidden_layers_as_outputs 的输出。

    seq 方向收集: 12 个 seq LayerNorm 输出 + output-seq softmax
      -> (batch, seq_len, 12*128 + 26) = (batch, seq_len, 1562)
    global 方向收集: dense-global-input + 12 个 global LayerNorm 输出 + output-annotations sigmoid
      -> (batch, 512 + 12*512 + 8943) = (batch, 15599)
    """
    seq_outputs = []
    global_outputs = []

    hidden_seq = model.seq_embedding(input_seq)
    # 注意: TF 版不收集 embedding 层 (名字不匹配过滤条件)

    hidden_global = F.gelu(model.global_input_dense(input_annotations))
    global_outputs.append(hidden_global)  # TF 版收集 dense-global-input

    for block in model.blocks:
        seqed_global = F.gelu(block.global_to_seq_dense(hidden_global)).unsqueeze(1)
        seq_t = hidden_seq.transpose(1, 2)
        narrow = F.gelu(block.narrow_conv(seq_t)).transpose(1, 2)
        wide = F.gelu(block.wide_conv(seq_t)).transpose(1, 2)
        hidden_seq = block.seq_norm1(hidden_seq + seqed_global + narrow + wide)
        seq_outputs.append(hidden_seq)  # seq-merge1-norm (LayerNorm, 3D)

        hidden_seq = block.seq_norm2(hidden_seq + F.gelu(block.seq_dense(hidden_seq)))
        seq_outputs.append(hidden_seq)  # seq-merge2-norm (LayerNorm, 3D)

        dense_g = F.gelu(block.global_dense1(hidden_global))
        attn = block.global_attention(hidden_global, hidden_seq)
        hidden_global = block.global_norm1(hidden_global + dense_g + attn)
        global_outputs.append(hidden_global)  # global-merge1-norm (LayerNorm, 2D)

        hidden_global = block.global_norm2(hidden_global + F.gelu(block.global_dense2(hidden_global)))
        global_outputs.append(hidden_global)  # global-merge2-norm (LayerNorm, 2D)

    output_seq = F.softmax(model.output_seq_dense(hidden_seq), dim=-1)
    seq_outputs.append(output_seq)  # output-seq

    output_annotations = torch.sigmoid(model.output_annotations_dense(hidden_global))
    global_outputs.append(output_annotations)  # output-annotations

    local_representations = torch.cat(seq_outputs, dim=-1)
    global_representations = torch.cat(global_outputs, dim=-1)

    return local_representations, global_representations


def encode_X(seqs, seq_len, n_annotations, device):
    token_ids = tokenize_seqs(seqs, seq_len)
    input_seq = torch.from_numpy(token_ids).long().to(device)
    input_annotations = torch.zeros(len(seqs), n_annotations, dtype=torch.float32).to(device)
    return input_seq, input_annotations


if __name__ == "__main__":

    print("=" * 70)
    print("ProteinBERT Embedding 提取 - PyTorch NPU 版")
    print("=" * 70)

    print("\n[1] Loading pretrained model...")
    pretrained_model, n_annotations = convert_tf_to_pytorch(PKL_PATH)
    pretrained_model = pretrained_model.to(DEVICE)
    pretrained_model.eval()
    print(f"    Model loaded on {DEVICE}")

    seqs = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "ACDEFGHIKLMNPQRSTUVWXY",
    ]
    seq_len = 512
    batch_size = 32

    print("\n[2] Extracting embeddings...")
    input_seq, input_annotations = encode_X(seqs, seq_len, n_annotations, DEVICE)

    with torch.no_grad():
        local_representations, global_representations = get_all_hidden_representations(
            pretrained_model, input_seq, input_annotations)

    local_np = local_representations.cpu().numpy()
    global_np = global_representations.cpu().numpy()

    print(f"\n[3] Results:")
    print(f"    local_representations  shape: {local_np.shape}")
    print(f"    global_representations shape: {global_np.shape}")
    print(f"    local_representations  dtype: {local_np.dtype}")
    print(f"    global_representations dtype: {global_np.dtype}")

    n_seq_layers = 6 * 2 + 1  # 12 LayerNorm + 1 output-seq
    n_global_layers = 1 + 6 * 2 + 1  # 1 input_dense + 12 LayerNorm + 1 output-annotations
    print(f"\n    拼接层数:")
    print(f"      seq 方向: {n_seq_layers} 层 (12 LayerNorm + 1 output-seq)")
    print(f"      global 方向: {n_global_layers} 层 (1 input_dense + 12 LayerNorm + 1 output)")
    print(f"      seq 维度拆解: 12 * 128 + 26 = {12 * 128 + 26}")
    print(f"      global 维度拆解: 13 * 512 + 8943 = {13 * 512 + 8943}")

    print(f"\n    每条序列的 embedding 示例:")
    for i, seq in enumerate(seqs):
        print(f"      Seq {i} (len={len(seq)}): "
              f"global_repr[:5] = {global_np[i, :5]}")

    print(f"\n    数值统计:")
    print(f"      local  mean={local_np.mean():.6f}, "
          f"std={local_np.std():.6f}, "
          f"min={local_np.min():.6f}, "
          f"max={local_np.max():.6f}")
    print(f"      global mean={global_np.mean():.6f}, "
          f"std={global_np.std():.6f}, "
          f"min={global_np.min():.6f}, "
          f"max={global_np.max():.6f}")

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "embeddings_npu.npz")
    np.savez(output_file,
             local_representations=local_np,
             global_representations=global_np)
    print(f"\n    结果已保存到: {output_file}")
    print(f"    (可用 compare_embeddings.py 与 GPU 结果对比)")

    sep = "=" * 70
    print(f"\n{sep}")
    print("Done!")
    print("=" * 70)
