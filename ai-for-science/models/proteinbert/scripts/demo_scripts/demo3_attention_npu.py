import os, sys, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from proteinbert_pytorch.convert_weights import convert_tf_to_pytorch
from proteinbert_pytorch.inference import tokenize_seqs

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protein_benchmarks')
PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')
DEVICE = 'npu:0'

TEST_SET_FILE_PATH = os.path.join(BENCHMARKS_DIR, 'signalP_binary.train.csv')
IDEAL_LEN = 80

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
index_to_token = {i: aa for i, aa in enumerate(ALL_AAS)}
index_to_token.update({i + len(ALL_AAS): tok for i, tok in enumerate(ADDITIONAL_TOKENS)})

def calculate_attentions(model, seq, seq_len, n_ann, device):
    tokens = tokenize_seqs([seq], seq_len)
    inp_seq = torch.from_numpy(tokens).long().to(device)
    inp_ann = torch.zeros(1, n_ann, dtype=torch.float32).to(device)
    seq_tokens = [index_to_token.get(t, '?') for t in tokens[0]]

    all_attns = []
    labels = []

    with torch.no_grad():
        hidden_seq = model.seq_embedding(inp_seq)
        hidden_global = F.gelu(model.global_input_dense(inp_ann))

        for bi, block in enumerate(model.blocks):
            seqed_global = F.gelu(block.global_to_seq_dense(hidden_global)).unsqueeze(1)
            seq_t = hidden_seq.transpose(1, 2)
            narrow = F.gelu(block.narrow_conv(seq_t)).transpose(1, 2)
            wide = F.gelu(block.wide_conv(seq_t)).transpose(1, 2)
            hidden_seq = block.seq_norm1(hidden_seq + seqed_global + narrow + wide)
            hidden_seq = block.seq_norm2(hidden_seq + F.gelu(block.seq_dense(hidden_seq)))

            attn_mod = block.global_attention
            batch_size = hidden_global.size(0)
            length = hidden_seq.size(1)

            QX = torch.tanh(torch.einsum('bg,hgk->bhk', hidden_global, attn_mod.Wq))
            QX = QX.reshape(batch_size * attn_mod.n_heads, attn_mod.d_key)
            KS = torch.tanh(torch.einsum('bls,hsk->bhkl', hidden_seq, attn_mod.Wk))
            KS = KS.reshape(batch_size * attn_mod.n_heads, attn_mod.d_key, length)
            attn_weights = F.softmax(torch.bmm(QX.unsqueeze(1), KS).squeeze(1) / attn_mod.sqrt_d_key, dim=-1)

            aw = attn_weights.cpu().numpy()
            for hi in range(attn_mod.n_heads):
                labels.append('Attention %d - head %d' % (bi + 1, hi + 1))
                all_attns.append(aw[hi])

            VS = F.gelu(torch.einsum('bls,hsv->bhlv', hidden_seq, attn_mod.Wv))
            VS = VS.reshape(batch_size * attn_mod.n_heads, length, attn_mod.d_value)
            Y = torch.bmm(attn_weights.unsqueeze(1), VS).squeeze(1)
            Y = Y.reshape(batch_size, attn_mod.d_output)
            dense_g = F.gelu(block.global_dense1(hidden_global))
            hidden_global = block.global_norm1(hidden_global + dense_g + Y)
            hidden_global = block.global_norm2(hidden_global + F.gelu(block.global_dense2(hidden_global)))

    return np.array(all_attns), seq_tokens, labels


test_set = pd.read_csv(TEST_SET_FILE_PATH)
chosen_index = ((test_set['seq'].str.len() - IDEAL_LEN).abs()).sort_values().index[0]
seq = test_set.loc[chosen_index, 'seq']
label = test_set.loc[chosen_index, 'label']

seq_len = len(seq) + 2

model, n_ann = convert_tf_to_pytorch(PKL_PATH)
model = model.to(DEVICE).eval()
attn_vals, seq_tokens, attn_labels = calculate_attentions(model, seq, seq_len, n_ann, DEVICE)

print(seq, label)

# ---- 日志：对标 GPU 版的打印格式 ----
print(f'\n[Compare Log] Sequence len={len(seq)}, seq_len={seq_len}')
print(f'[Compare Log] Attention shape: {attn_vals.shape}')
for i, lbl in enumerate(attn_labels):
    v = attn_vals[i]
    print(f'[Compare Log] {lbl:30s} mean={v.mean():.6f} std={v.std():.6f} max={v.max():.6f}')
total_attn = attn_vals.sum(axis=0)
print(f'[Compare Log] Total attention per position (first 10):')
for j in range(min(10, len(seq_tokens))):
    print(f'  pos {j:3d} [{seq_tokens[j]:>7s}]: {total_attn[j]:.6f}')

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'attention_npu.npz'),
         attention_values=attn_vals, seq_tokens=np.array(seq_tokens), attention_labels=np.array(attn_labels))
print(f'\n[Compare Log] Saved to attention_npu.npz')
