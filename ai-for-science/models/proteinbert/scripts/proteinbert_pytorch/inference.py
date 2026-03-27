import os
import pickle
import numpy as np
import torch

from .model import ProteinBERTModel

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
n_tokens = len(aa_to_token_index) + len(additional_token_to_index)


def tokenize_seq(seq):
    other_idx = additional_token_to_index['<OTHER>']
    return ([additional_token_to_index['<START>']] +
            [aa_to_token_index.get(aa, other_idx) for aa in seq] +
            [additional_token_to_index['<END>']])


def tokenize_seqs(seqs, seq_len):
    pad_idx = additional_token_to_index['<PAD>']
    result = []
    for seq in seqs:
        tokens = tokenize_seq(seq)
        tokens = tokens + [pad_idx] * (seq_len - len(tokens))
        result.append(tokens[:seq_len])
    return np.array(result, dtype=np.int32)


def load_pretrained_model_pt(pkl_path, device='cpu'):
    with open(pkl_path, 'rb') as f:
        n_annotations, model_weights, _ = pickle.load(f)

    from .convert_weights import convert_tf_to_pytorch
    model, n_annotations = convert_tf_to_pytorch(pkl_path)
    model = model.to(device)
    model.eval()
    return model, n_annotations


def predict_embeddings(model, seqs, seq_len=512, batch_size=32, device='cpu'):
    token_ids = tokenize_seqs(seqs, seq_len)
    n_annotations = model.n_annotations
    all_seq_out = []
    all_global_out = []

    with torch.no_grad():
        for start in range(0, len(token_ids), batch_size):
            batch_tokens = token_ids[start:start + batch_size]
            input_seq = torch.from_numpy(batch_tokens).long().to(device)
            input_annots = torch.zeros(len(batch_tokens), n_annotations,
                                       dtype=torch.float32).to(device)
            out_seq, out_annots = model(input_seq, input_annots)
            all_seq_out.append(out_seq.cpu().numpy())
            all_global_out.append(out_annots.cpu().numpy())

    return np.concatenate(all_seq_out, axis=0), np.concatenate(all_global_out, axis=0)
