#!/usr/bin/env python3
"""
ProteinBERT Embedding 提取 - 原始 TF/Keras GPU 版

使用原始 TensorFlow 实现，打印与 NPU 版相同的信息便于对比。

用法:
    pip install tensorflow==2.4.0 tensorflow_addons numpy pandas h5py lxml pyfaidx
    cd protein_bert && python setup.py install
    python get_embeddings_gpu.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PKL_PATH = os.path.expanduser('~/proteinbert_models/epoch_92400_sample_23500000.pkl')


if __name__ == '__main__':

    print('=' * 70)
    print('ProteinBERT Embedding 提取 - 原始 TF/Keras GPU 版')
    print('=' * 70)

    # ---- 加载模型 ----
    print('\n[1] Loading pretrained model...')
    from proteinbert import load_pretrained_model
    from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

    pretrained_model_generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=os.path.dirname(PKL_PATH),
        local_model_dump_file_name=os.path.basename(PKL_PATH),
        download_model_dump_if_not_exists=False,
    )
    print(f'    n_annotations: {input_encoder.n_annotations}')

    seqs = [
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE',
        'ACDEFGHIKLMNPQRSTUVWXY',
    ]
    seq_len = 512
    batch_size = 32

    # ---- 创建模型并提取隐藏层 ----
    print('\n[2] Extracting embeddings...')
    model = get_model_with_hidden_layers_as_outputs(
        pretrained_model_generator.create_model(seq_len))

    encoded_x = input_encoder.encode_X(seqs, seq_len)
    local_representations, global_representations = model.predict(
        encoded_x, batch_size=batch_size)

    # ---- 打印结果（与 NPU 版格式完全一致） ----
    print(f'\n[3] Results:')
    print(f'    local_representations  shape: {local_representations.shape}')
    print(f'    global_representations shape: {global_representations.shape}')
    print(f'    local_representations  dtype: {local_representations.dtype}')
    print(f'    global_representations dtype: {global_representations.dtype}')

    n_seq_layers = 6 * 2 + 1
    n_global_layers = 1 + 6 * 2 + 1
    print(f'\n    拼接层数:')
    print(f'      seq 方向: {n_seq_layers} 层 (12 LayerNorm + 1 output-seq)')
    print(f'      global 方向: {n_global_layers} 层 (1 input_dense + 12 LayerNorm + 1 output)')
    print(f'      seq 维度拆解: 12 * 128 + 26 = {12 * 128 + 26}')
    print(f'      global 维度拆解: 13 * 512 + 8943 = {13 * 512 + 8943}')

    print(f'\n    每条序列的 embedding 示例:')
    for i, seq in enumerate(seqs):
        print(f'      Seq {i} (len={len(seq)}): '
              f'global_repr[:5] = {global_representations[i, :5]}')

    # ---- 数值统计（便于对比） ----
    print(f'\n    数值统计:')
    print(f'      local  mean={local_representations.mean():.6f}, '
          f'std={local_representations.std():.6f}, '
          f'min={local_representations.min():.6f}, '
          f'max={local_representations.max():.6f}')
    print(f'      global mean={global_representations.mean():.6f}, '
          f'std={global_representations.std():.6f}, '
          f'min={global_representations.min():.6f}, '
          f'max={global_representations.max():.6f}')

    # ---- 保存结果便于精确对比 ----
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'embeddings_gpu.npz')
    np.savez(output_file,
             local_representations=local_representations,
             global_representations=global_representations)
    print(f'\n    结果已保存到: {output_file}')
    print(f'    (可用 compare_embeddings.py 与 NPU 结果对比)')

    sep = '=' * 70
    print(f'\n{sep}')
    print('Done!')
    print('=' * 70)
