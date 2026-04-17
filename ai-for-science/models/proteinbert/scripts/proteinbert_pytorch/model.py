import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        VS = torch.einsum('bls,hsv->bhlv', seq_repr, self.Wv)
        VS = F.gelu(VS)
        VS = VS.reshape(batch_size * self.n_heads, length, self.d_value)

        QX = torch.einsum('bg,hgk->bhk', global_repr, self.Wq)
        QX = torch.tanh(QX)
        QX = QX.reshape(batch_size * self.n_heads, self.d_key)

        KS = torch.einsum('bls,hsk->bhkl', seq_repr, self.Wk)
        KS = torch.tanh(KS)
        KS = KS.reshape(batch_size * self.n_heads, self.d_key, length)

        attn_scores = torch.bmm(QX.unsqueeze(1), KS).squeeze(1) / self.sqrt_d_key
        attn_weights = F.softmax(attn_scores, dim=-1)

        Y = torch.bmm(attn_weights.unsqueeze(1), VS).squeeze(1)
        Y = Y.reshape(batch_size, self.d_output)
        return Y


class ProteinBERTBlock(nn.Module):
    def __init__(self, d_hidden_seq, d_hidden_global, n_heads, d_key,
                 conv_kernel_size, wide_conv_dilation_rate):
        super().__init__()
        d_value = d_hidden_global // n_heads

        self.global_to_seq_dense = nn.Linear(d_hidden_global, d_hidden_seq)
        self.narrow_conv = nn.Conv1d(d_hidden_seq, d_hidden_seq,
                                     kernel_size=conv_kernel_size,
                                     padding='same', dilation=1)
        self.wide_conv = nn.Conv1d(d_hidden_seq, d_hidden_seq,
                                   kernel_size=conv_kernel_size,
                                   padding='same',
                                   dilation=wide_conv_dilation_rate)
        self.seq_norm1 = nn.LayerNorm(d_hidden_seq, eps=1e-3)
        self.seq_dense = nn.Linear(d_hidden_seq, d_hidden_seq)
        self.seq_norm2 = nn.LayerNorm(d_hidden_seq, eps=1e-3)

        self.global_dense1 = nn.Linear(d_hidden_global, d_hidden_global)
        self.global_attention = GlobalAttention(n_heads, d_key, d_value,
                                               d_hidden_global, d_hidden_seq)
        self.global_norm1 = nn.LayerNorm(d_hidden_global, eps=1e-3)
        self.global_dense2 = nn.Linear(d_hidden_global, d_hidden_global)
        self.global_norm2 = nn.LayerNorm(d_hidden_global, eps=1e-3)

    def forward(self, hidden_seq, hidden_global):
        seqed_global = F.gelu(self.global_to_seq_dense(hidden_global))
        seqed_global = seqed_global.unsqueeze(1)

        seq_t = hidden_seq.transpose(1, 2)
        narrow = F.gelu(self.narrow_conv(seq_t)).transpose(1, 2)
        wide = F.gelu(self.wide_conv(seq_t)).transpose(1, 2)

        hidden_seq = hidden_seq + seqed_global + narrow + wide
        hidden_seq = self.seq_norm1(hidden_seq)

        dense_seq = F.gelu(self.seq_dense(hidden_seq))
        hidden_seq = hidden_seq + dense_seq
        hidden_seq = self.seq_norm2(hidden_seq)

        dense_global = F.gelu(self.global_dense1(hidden_global))
        attention = self.global_attention(hidden_global, hidden_seq)
        hidden_global = hidden_global + dense_global + attention
        hidden_global = self.global_norm1(hidden_global)

        dense_global2 = F.gelu(self.global_dense2(hidden_global))
        hidden_global = hidden_global + dense_global2
        hidden_global = self.global_norm2(hidden_global)

        return hidden_seq, hidden_global


class ProteinBERTModel(nn.Module):
    def __init__(self, vocab_size=26, n_annotations=8943,
                 d_hidden_seq=128, d_hidden_global=512,
                 n_blocks=6, n_heads=4, d_key=64,
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

        seq_hiddens = []
        global_hiddens = []

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
