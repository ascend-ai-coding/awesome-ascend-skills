#!/usr/bin/env python3
"""
ProteinBERT NPU 推理脚本（独立脚本）

加载转换后的 .pt 权重，在昇腾 NPU 上运行推理。

用法:
    python inference_npu.py \
        --weights ~/proteinbert_models/proteinbert_pytorch.pt \
        --seqs "MKTVRQERLK" "ACDEFGHIKL" \
        --device npu:0

或从文件读取序列（每行一条）:
    python inference_npu.py \
        --weights ~/proteinbert_models/proteinbert_pytorch.pt \
        --seqs-file sequences.txt \
        --device npu:0
"""
import argparse
import os
import numpy as np
import torch

# ---- Tokenizer ----
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
aa_to_idx = {aa: i for i, aa in enumerate(ALL_AAS)}
extra_to_idx = {tok: i + len(ALL_AAS) for i, tok in enumerate(ADDITIONAL_TOKENS)}


def tokenize_seqs(seqs, seq_len):
    pad = extra_to_idx['<PAD>']
    start = extra_to_idx['<START>']
    end = extra_to_idx['<END>']
    other = extra_to_idx['<OTHER>']
    result = []
    for seq in seqs:
        tokens = [start] + [aa_to_idx.get(aa, other) for aa in seq] + [end]
        tokens = (tokens + [pad] * seq_len)[:seq_len]
        result.append(tokens)
    return np.array(result, dtype=np.int32)


def load_model(pt_path, device='cpu'):
    from convert_weights import ProteinBERTModel
    ckpt = torch.load(pt_path, map_location='cpu')
    model = ProteinBERTModel(
        vocab_size=ckpt['vocab_size'],
        n_annotations=ckpt['n_annotations'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    return model, ckpt['n_annotations']


def predict(model, seqs, n_annotations, seq_len=512, batch_size=32, device='cpu'):
    token_ids = tokenize_seqs(seqs, seq_len)
    all_seq, all_global = [], []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            inp_seq = torch.from_numpy(batch).long().to(device)
            inp_ann = torch.zeros(len(batch), n_annotations, dtype=torch.float32).to(device)
            out_seq, out_ann = model(inp_seq, inp_ann)
            all_seq.append(out_seq.cpu().numpy())
            all_global.append(out_ann.cpu().numpy())
    return np.concatenate(all_seq), np.concatenate(all_global)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProteinBERT NPU 推理')
    parser.add_argument('--weights', '-w', required=True, help='转换后的 .pt 文件')
    parser.add_argument('--seqs', nargs='+', default=None, help='蛋白质序列（空格分隔）')
    parser.add_argument('--seqs-file', default=None, help='序列文件（每行一条）')
    parser.add_argument('--seq-len', type=int, default=512, help='序列长度 (default: 512)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='npu:0', help='设备 (npu:0 / cpu)')
    parser.add_argument('--output', '-o', default=None, help='输出 npz 文件路径')
    args = parser.parse_args()

    if args.device.startswith('npu'):
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

    seqs = args.seqs or []
    if args.seqs_file:
        with open(args.seqs_file) as f:
            seqs.extend([line.strip() for line in f if line.strip()])
    if not seqs:
        seqs = [
            'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE',
        ]
        print(f'未指定序列，使用 {len(seqs)} 条示例序列')

    print(f'加载模型: {args.weights}')
    model, n_ann = load_model(args.weights, args.device)
    print(f'设备: {args.device}, 序列数: {len(seqs)}')

    out_seq, out_global = predict(model, seqs, n_ann, args.seq_len, args.batch_size, args.device)
    print(f'输出: seq_probs {out_seq.shape}, annotations {out_global.shape}')

    for i, seq in enumerate(seqs[:5]):
        top5 = np.sort(out_global[i])[-5:][::-1]
        print(f'  Seq {i} (len={len(seq)}): top annotation scores = {top5}')

    if args.output:
        np.savez(args.output, seq_probs=out_seq, annotation_scores=out_global)
        print(f'结果已保存到: {args.output}')
