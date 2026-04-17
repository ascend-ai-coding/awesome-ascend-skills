import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protein_benchmarks')
PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')
DEVICE = 'npu:0'

# ---- Dataset ----
class ProteinDS(Dataset):
    def __init__(self, toks, anns, lbls):
        self.toks = torch.from_numpy(toks).long()
        self.anns = torch.from_numpy(anns).float()
        self.lbls = torch.from_numpy(lbls).float()
    def __len__(self): return len(self.lbls)
    def __getitem__(self, i): return self.toks[i], self.anns[i], self.lbls[i]

# ---- FTModel: 对标 TF FinetuningModelGenerator + get_model_with_hidden_layers_as_outputs ----
# TF 版在微调时，先用 get_model_with_hidden_layers_as_outputs 把模型输出改为
# 所有隐藏层拼接，然后从拼接后的 global 输出 (15599维) 接 Dropout + Dense head
# global 15599 = dense-global-input(512) + 12*LayerNorm(512) + output-annotations(8943)
class FTModel(nn.Module):
    def __init__(self, base, dropout=0.5):
        super().__init__()
        self.base = base
        self.drop = nn.Dropout(dropout)
        global_concat_dim = 13 * base.d_hidden_global + base.n_annotations  # 15599
        self.head = nn.Linear(global_concat_dim, 1)

    def forward(self, seq, ann):
        h_s = self.base.seq_embedding(seq)
        h_g = F.gelu(self.base.global_input_dense(ann))
        global_outputs = [h_g]
        for block in self.base.blocks:
            seqed_global = F.gelu(block.global_to_seq_dense(h_g)).unsqueeze(1)
            seq_t = h_s.transpose(1, 2)
            narrow = F.gelu(block.narrow_conv(seq_t)).transpose(1, 2)
            wide = F.gelu(block.wide_conv(seq_t)).transpose(1, 2)
            h_s = block.seq_norm1(h_s + seqed_global + narrow + wide)
            h_s = block.seq_norm2(h_s + F.gelu(block.seq_dense(h_s)))
            dense_g = F.gelu(block.global_dense1(h_g))
            attn = block.global_attention(h_g, h_s)
            h_g = block.global_norm1(h_g + dense_g + attn)
            global_outputs.append(h_g)
            h_g = block.global_norm2(h_g + F.gelu(block.global_dense2(h_g)))
            global_outputs.append(h_g)
        out_ann = torch.sigmoid(self.base.output_annotations_dense(h_g))
        global_outputs.append(out_ann)
        concat_global = torch.cat(global_outputs, dim=-1)
        return self.head(self.drop(concat_global)).squeeze(-1)

def mkdl(seqs, labels, n_ann, sl, bs, shuf=False):
    t = tokenize_seqs(list(seqs), sl)
    a = np.zeros((len(seqs), n_ann), dtype=np.float32)
    l = np.array(list(labels), dtype=np.float32)
    return DataLoader(ProteinDS(t, a, l), batch_size=bs, shuffle=shuf, num_workers=0)

def train_ep(m, dl, opt, dev):
    m.train(); tl, n = 0., 0
    for tk, an, lb in dl:
        tk, an, lb = tk.to(dev), an.to(dev), lb.to(dev)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(tk, an), lb)
        loss.backward(); opt.step()
        tl += loss.item() * len(lb); n += len(lb)
    return tl / n

def eval_fn(m, dl, dev):
    m.eval(); ps, ls, tl, n = [], [], 0., 0
    with torch.no_grad():
        for tk, an, lb in dl:
            tk, an, lb = tk.to(dev), an.to(dev), lb.to(dev)
            lo = m(tk, an); loss = F.binary_cross_entropy_with_logits(lo, lb)
            ps.append(torch.sigmoid(lo).cpu().numpy()); ls.append(lb.cpu().numpy())
            tl += loss.item() * len(lb); n += len(lb)
    return tl / n, np.concatenate(ps), np.concatenate(ls)

def run_stage(ft, trdl, vdl, dev, epochs, lr, patience, freeze=False):
    if freeze:
        for p in ft.base.parameters(): p.requires_grad = False
        ft.head.requires_grad_(True); ft.drop.requires_grad_(True)
    else:
        for p in ft.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, ft.parameters()), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, factor=0.25, min_lr=1e-5)
    best, bst, w = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        tl = train_ep(ft, trdl, opt, dev)
        vl, vp, vy = eval_fn(ft, vdl, dev); sch.step(vl)
        tag = ' [frozen]' if freeze else ''
        try: auc_s = f', val_auc: {roc_auc_score(vy, vp):.4f}'
        except: auc_s = ''
        print(f'  Epoch {ep}/{epochs}{tag} - train_loss: {tl:.4f}, val_loss: {vl:.4f}{auc_s}')
        if vl < best:
            best = vl; bst = {k: v.clone() for k, v in ft.state_dict().items()}; w = 0
        else:
            w += 1
            if w >= patience: print(f'  Early stopping at epoch {ep}'); break
    if bst: ft.load_state_dict(bst)


# ---- Main: 对标 demo notebook Cell 3 ----

BENCHMARK_NAME = 'signalP_binary'

# Loading the dataset (与 TF 版完全相同)
train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % BENCHMARK_NAME)
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
train_set, valid_set = train_test_split(train_set, stratify=train_set['label'], test_size=0.1, random_state=0)

test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % BENCHMARK_NAME)
test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

# Loading the pre-trained model
pretrained_model, n_ann = convert_tf_to_pytorch(PKL_PATH)
pretrained_model = pretrained_model.to(DEVICE)

ft = FTModel(pretrained_model, dropout=0.5).to(DEVICE)
trdl = mkdl(train_set['seq'], train_set['label'], n_ann, 512, 32, True)
vdl = mkdl(valid_set['seq'], valid_set['label'], n_ann, 512, 32)

# 对标: begin_with_frozen_pretrained_layers=True, lr_with_frozen_pretrained_layers=1e-02
print('\nTraining with frozen pretrained layers...')
run_stage(ft, trdl, vdl, DEVICE, 40, 1e-02, 2, freeze=True)

# 对标: lr=1e-04
print('\nTraining the entire fine-tuned model...')
run_stage(ft, trdl, vdl, DEVICE, 40, 1e-04, 2, freeze=False)

# 对标: n_final_epochs=1, final_seq_len=1024, final_lr=1e-05
print('\nTraining on final epochs of sequence length 1024...')
trdl_f = mkdl(train_set['seq'], train_set['label'], n_ann, 1024, 16, True)
vdl_f = mkdl(valid_set['seq'], valid_set['label'], n_ann, 1024, 16)
run_stage(ft, trdl_f, vdl_f, DEVICE, 1, 1e-05, 2, freeze=False)

# Evaluating the performance on the test-set
print('\nTest-set performance:')
tedl = mkdl(test_set['seq'], test_set['label'], n_ann, 512, 32)
_, tp, ty = eval_fn(ft, tedl, DEVICE)
auc = roc_auc_score(ty, tp); acc = accuracy_score(ty, (tp > 0.5).astype(int))
from sklearn.metrics import confusion_matrix as sk_cm
cm = sk_cm(ty, (tp > 0.5).astype(int))
print(f'  AUC: {auc:.6f}, Accuracy: {acc:.6f}')
print(f'Confusion matrix:')
print(cm)
print(f'\n[Compare Log] Test AUC: {auc:.6f}, Test Acc: {acc:.6f}')
