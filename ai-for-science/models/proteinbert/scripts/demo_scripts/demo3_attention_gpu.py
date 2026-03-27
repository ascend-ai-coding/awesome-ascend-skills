import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protein_benchmarks')
TEST_SET_FILE_PATH = os.path.join(BENCHMARKS_DIR, 'signalP_binary.train.csv')
IDEAL_LEN = 80

from tensorflow.keras import backend as K
from proteinbert import load_pretrained_model
from proteinbert.tokenization import index_to_token

def calculate_attentions(model, input_encoder, seq, seq_len = None):

    if seq_len is None:
        seq_len = len(seq) + 2

    X = input_encoder.encode_X([seq], seq_len)
    (X_seq,), _ = X
    seq_tokens = list(map(index_to_token.get, X_seq))

    model_inputs = [layer.input for layer in model.layers if 'InputLayer' in str(type(layer))][::-1]
    model_attentions = [layer.calculate_attention(layer.input) for layer in model.layers if 'GlobalAttention' in str(type(layer))]
    invoke_model_attentions = K.function(model_inputs, model_attentions)
    attention_values = invoke_model_attentions(X)

    attention_labels = []
    merged_attention_values = []

    for attention_layer_index, attention_layer_values in enumerate(attention_values):
        for head_index, head_values in enumerate(attention_layer_values):
            attention_labels.append('Attention %d - head %d' % (attention_layer_index + 1, head_index + 1))
            merged_attention_values.append(head_values)

    attention_values = np.array(merged_attention_values)

    return attention_values, seq_tokens, attention_labels


test_set = pd.read_csv(TEST_SET_FILE_PATH)
chosen_index = ((test_set['seq'].str.len() - IDEAL_LEN).abs()).sort_values().index[0]
seq = test_set.loc[chosen_index, 'seq']
label = test_set.loc[chosen_index, 'label']

seq_len = len(seq) + 2

pretrained_model_generator, input_encoder = load_pretrained_model()
model = pretrained_model_generator.create_model(seq_len)
attn_vals, seq_tokens, attn_labels = calculate_attentions(model, input_encoder, seq, seq_len = seq_len)

print(seq, label)

# ---- 日志：打印 attention 统计便于对比 ----
print(f'\n[Compare Log] Sequence len={len(seq)}, seq_len={seq_len}')
print(f'[Compare Log] Attention shape: {attn_vals.shape}')
for i, lbl in enumerate(attn_labels):
    v = attn_vals[i]
    print(f'[Compare Log] {lbl:30s} mean={v.mean():.6f} std={v.std():.6f} max={v.max():.6f}')
total_attn = attn_vals.sum(axis=0)
print(f'[Compare Log] Total attention per position (first 10):')
for j in range(min(10, len(seq_tokens))):
    print(f'  pos {j:3d} [{seq_tokens[j]:>7s}]: {total_attn[j]:.6f}')

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'attention_gpu.npz'),
         attention_values=attn_vals, seq_tokens=np.array(seq_tokens), attention_labels=np.array(attn_labels))
print(f'\n[Compare Log] Saved to attention_gpu.npz')
