#!/usr/bin/env python3
"""
ProteinBERT NPU 微调脚本（独立脚本）

加载转换后的 .pt 权重，在昇腾 NPU 上进行微调训练。

用法:
    python finetune_npu.py \
        --weights ~/proteinbert_models/proteinbert_pytorch.pt \
        --train-csv protein_benchmarks/signalP_binary.train.csv \
        --test-csv protein_benchmarks/signalP_binary.test.csv \
        --task binary \
        --device npu:0
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


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


# ---- Dataset ----
class ProteinDataset(Dataset):
    def __init__(self, token_ids, annotations, labels):
        self.token_ids = torch.from_numpy(token_ids).long()
        self.annotations = torch.from_numpy(annotations).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.annotations[idx], self.labels[idx]


# ---- Finetune Head ----
class FinetuneModel(nn.Module):
    def __init__(self, pretrained, output_type='binary', n_classes=1,
                 use_global=True, dropout_rate=0.5):
        super().__init__()
        self.pretrained = pretrained
        self.use_global = use_global
        self.output_type = output_type
        hidden_dim = pretrained.d_hidden_global if use_global else pretrained.d_hidden_seq
        self.dropout = nn.Dropout(dropout_rate)
        if output_type == 'categorical':
            self.head = nn.Linear(hidden_dim, n_classes)
        else:
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_seq, input_annotations):
        hidden_seq = self.pretrained.seq_embedding(input_seq)
        hidden_global = F.gelu(self.pretrained.global_input_dense(input_annotations))
        for block in self.pretrained.blocks:
            hidden_seq, hidden_global = block(hidden_seq, hidden_global)
        features = hidden_global if self.use_global else hidden_seq
        logits = self.head(self.dropout(features))
        return logits.squeeze(-1) if self.output_type != 'categorical' else logits


# ---- Train / Eval ----
def train_epoch(model, loader, optimizer, device, output_type):
    model.train()
    total_loss, n = 0.0, 0
    for tok, ann, lbl in loader:
        tok, ann, lbl = tok.to(device), ann.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(tok, ann)
        if output_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
        elif output_type == 'categorical':
            loss = F.cross_entropy(logits, lbl.long())
        else:
            loss = F.mse_loss(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(lbl)
        n += len(lbl)
    return total_loss / n


def evaluate(model, loader, device, output_type):
    model.eval()
    preds, labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for tok, ann, lbl in loader:
            tok, ann, lbl = tok.to(device), ann.to(device), lbl.to(device)
            logits = model(tok, ann)
            if output_type == 'binary':
                loss = F.binary_cross_entropy_with_logits(logits, lbl)
                preds.append(torch.sigmoid(logits).cpu().numpy())
            elif output_type == 'categorical':
                loss = F.cross_entropy(logits, lbl.long())
                preds.append(logits.argmax(-1).float().cpu().numpy())
            else:
                loss = F.mse_loss(logits, lbl)
                preds.append(logits.cpu().numpy())
            labels.append(lbl.cpu().numpy())
            total_loss += loss.item() * len(lbl)
            n += len(lbl)
    return total_loss / n, np.concatenate(preds), np.concatenate(labels)


def make_loader(seqs, labels, n_ann, seq_len, batch_size, shuffle=False):
    tok = tokenize_seqs(list(seqs), seq_len)
    ann = np.zeros((len(seqs), n_ann), dtype=np.float32)
    lbl = np.array(list(labels), dtype=np.float32)
    return DataLoader(ProteinDataset(tok, ann, lbl), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def run_stage(ft_model, train_dl, valid_dl, device, output_type,
              epochs, lr, patience, freeze=False):
    if freeze:
        for p in ft_model.pretrained.parameters():
            p.requires_grad = False
        ft_model.head.requires_grad_(True)
    else:
        for p in ft_model.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ft_model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=0.25, min_lr=1e-5)

    best_loss, best_state, wait = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(ft_model, train_dl, optimizer, device, output_type)
        val_loss, val_preds, val_labels = evaluate(ft_model, valid_dl, device, output_type)
        scheduler.step(val_loss)
        tag = ' [frozen]' if freeze else ''
        msg = f'  Epoch {epoch}/{epochs}{tag} - train: {train_loss:.4f}, val: {val_loss:.4f}'
        if output_type == 'binary':
            try:
                msg += f', auc: {roc_auc_score(val_labels, val_preds):.4f}'
            except ValueError:
                pass
        print(msg)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'  Early stopping at epoch {epoch}')
                break
    if best_state:
        ft_model.load_state_dict(best_state)


# ---- Main ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProteinBERT NPU 微调')
    parser.add_argument('--weights', '-w', required=True, help='转换后的 .pt 文件')
    parser.add_argument('--train-csv', required=True, help='训练集 CSV (seq, label 列)')
    parser.add_argument('--test-csv', required=True, help='测试集 CSV')
    parser.add_argument('--task', default='binary', choices=['binary', 'categorical', 'numeric'])
    parser.add_argument('--n-classes', type=int, default=2, help='类别数 (多分类时)')
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--frozen-lr', type=float, default=1e-2)
    parser.add_argument('--final-lr', type=float, default=1e-5)
    parser.add_argument('--final-seq-len', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--save-model', default=None, help='保存微调模型路径')
    args = parser.parse_args()

    if args.device.startswith('npu'):
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

    print('=' * 60)
    print('ProteinBERT NPU Fine-tuning')
    print('=' * 60)

    print(f'\n[1] 加载模型: {args.weights}')
    from convert_weights import ProteinBERTModel
    ckpt = torch.load(args.weights, map_location='cpu')
    model = ProteinBERTModel(vocab_size=ckpt['vocab_size'], n_annotations=ckpt['n_annotations'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(args.device)
    n_ann = ckpt['n_annotations']
    print(f'    已加载到 {args.device}')

    print(f'\n[2] 加载数据集')
    train_df = pd.read_csv(args.train_csv).dropna().drop_duplicates()
    test_df = pd.read_csv(args.test_csv).dropna().drop_duplicates()
    train_df, valid_df = train_test_split(
        train_df, stratify=train_df['label'], test_size=0.1, random_state=0)
    print(f'    Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}')

    train_dl = make_loader(train_df['seq'], train_df['label'], n_ann, args.seq_len, args.batch_size, True)
    valid_dl = make_loader(valid_df['seq'], valid_df['label'], n_ann, args.seq_len, args.batch_size)

    ft = FinetuneModel(model, args.task, args.n_classes).to(args.device)

    print(f'\n[3] 微调训练')
    print('\nStage 1: 冻结预训练层训练...')
    run_stage(ft, train_dl, valid_dl, args.device, args.task,
              args.max_epochs, args.frozen_lr, args.patience, freeze=True)

    print('\nStage 2: 全层训练...')
    run_stage(ft, train_dl, valid_dl, args.device, args.task,
              args.max_epochs, args.lr, args.patience, freeze=False)

    print(f'\nStage 3: 长序列最终训练 (seq_len={args.final_seq_len})...')
    final_bs = max(int(args.batch_size / (args.final_seq_len / args.seq_len)), 1)
    train_dl_f = make_loader(train_df['seq'], train_df['label'], n_ann, args.final_seq_len, final_bs, True)
    valid_dl_f = make_loader(valid_df['seq'], valid_df['label'], n_ann, args.final_seq_len, final_bs)
    run_stage(ft, train_dl_f, valid_dl_f, args.device, args.task,
              1, args.final_lr, args.patience, freeze=False)

    print(f'\n[4] 测试集评估')
    test_dl = make_loader(test_df['seq'], test_df['label'], n_ann, args.seq_len, args.batch_size)
    test_loss, test_preds, test_labels = evaluate(ft, test_dl, args.device, args.task)

    print(f'\n{"=" * 60}')
    print(f'  Test Loss: {test_loss:.4f}')
    if args.task == 'binary':
        auc = roc_auc_score(test_labels, test_preds)
        acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
        cm = confusion_matrix(test_labels, (test_preds > 0.5).astype(int))
        print(f'  Test AUC:  {auc:.4f}')
        print(f'  Test Acc:  {acc:.4f}')
        print(f'  Confusion Matrix:\n    {cm}')
    print(f'{"=" * 60}')

    if args.save_model:
        torch.save(ft.state_dict(), args.save_model)
        print(f'微调模型已保存: {args.save_model}')
