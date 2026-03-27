#!/usr/bin/env python3
"""
逐层输出调试脚本 - GPU/TF 版
在 GPU 机器上运行，打印每层的 mean/std 便于与 NPU 对比。
"""
import os, sys, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')

from proteinbert import load_pretrained_model

pretrained_model_generator, input_encoder = load_pretrained_model(
    local_model_dump_dir=os.path.dirname(PKL_PATH),
    local_model_dump_file_name=os.path.basename(PKL_PATH),
    download_model_dump_if_not_exists=False,
)

seq_len = 512
model = pretrained_model_generator.create_model(seq_len)

seqs = [
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE',
    'ACDEFGHIKLMNPQRSTUVWXY',
]
encoded_x = input_encoder.encode_X(seqs, seq_len)

# 打印每层的名字和统计信息
interesting_layers = []
for layer in model.layers:
    name = layer.name
    if any(kw in name for kw in ['embedding', 'dense-global-input',
                                  'merge1-norm', 'merge2-norm',
                                  'output-seq', 'output-annotations']):
        interesting_layers.append(layer)

debug_model = keras.Model(inputs=model.inputs,
                          outputs=[l.output for l in interesting_layers])

results = debug_model.predict(encoded_x, batch_size=32)

for layer, output in zip(interesting_layers, results):
    arr = np.array(output)
    print(f'{layer.name:45s} shape={str(arr.shape):25s} mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}')
