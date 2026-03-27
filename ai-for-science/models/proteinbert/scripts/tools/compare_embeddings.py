#!/usr/bin/env python3
"""
对比 GPU 和 NPU 的 embedding 输出

用法:
    1. 在 GPU 机器上运行: python get_embeddings_gpu.py  → 生成 embeddings_gpu.npz
    2. 在 NPU 机器上运行: python get_embeddings_npu.py  → 生成 embeddings_npu.npz
    3. 将两个 npz 文件放到同一目录，运行: python compare_embeddings.py
"""
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_FILE = os.path.join(SCRIPT_DIR, 'embeddings_gpu.npz')
NPU_FILE = os.path.join(SCRIPT_DIR, 'embeddings_npu.npz')


def compare_arrays(name, gpu_arr, npu_arr):
    print(f'\n  {name}:')
    print(f'    shape   GPU={gpu_arr.shape}  NPU={npu_arr.shape}')
    if gpu_arr.shape != npu_arr.shape:
        print(f'    ⚠️  Shape 不一致！')
        return

    diff = np.abs(gpu_arr - npu_arr)
    rel_diff = diff / (np.abs(gpu_arr) + 1e-8)

    print(f'    绝对误差  mean={diff.mean():.2e}, max={diff.max():.2e}, '
          f'median={np.median(diff):.2e}')
    print(f'    相对误差  mean={rel_diff.mean():.2e}, max={rel_diff.max():.2e}, '
          f'median={np.median(rel_diff):.2e}')

    cos_sims = []
    for i in range(gpu_arr.shape[0]):
        g = gpu_arr[i].flatten()
        n = npu_arr[i].flatten()
        cos = np.dot(g, n) / (np.linalg.norm(g) * np.linalg.norm(n) + 1e-8)
        cos_sims.append(cos)
    print(f'    余弦相似度  {["{:.6f}".format(c) for c in cos_sims]}')

    threshold_1e3 = (diff < 1e-3).mean() * 100
    threshold_1e5 = (diff < 1e-5).mean() * 100
    print(f'    误差<1e-3: {threshold_1e3:.1f}%  误差<1e-5: {threshold_1e5:.1f}%')

    if diff.max() < 1e-3:
        print(f'    ✅ 精度对齐（fp32 场景差异 < 1e-3）')
    elif diff.max() < 1e-1:
        print(f'    ✅ 精度可接受（fp16/bf16 场景）')
    else:
        print(f'    ⚠️  误差较大，请检查权重转换或模型实现')


if __name__ == '__main__':
    print('=' * 60)
    print('ProteinBERT Embedding 对比: GPU vs NPU')
    print('=' * 60)

    if not os.path.exists(GPU_FILE):
        print(f'\n未找到 GPU 结果: {GPU_FILE}')
        print('请先在 GPU 机器上运行: python get_embeddings_gpu.py')
        exit(1)
    if not os.path.exists(NPU_FILE):
        print(f'\n未找到 NPU 结果: {NPU_FILE}')
        print('请先在 NPU 机器上运行: python get_embeddings_npu.py')
        exit(1)

    gpu_data = np.load(GPU_FILE)
    npu_data = np.load(NPU_FILE)

    print(f'\n文件:')
    print(f'  GPU: {GPU_FILE}')
    print(f'  NPU: {NPU_FILE}')

    compare_arrays('local_representations',
                   gpu_data['local_representations'],
                   npu_data['local_representations'])

    compare_arrays('global_representations',
                   gpu_data['global_representations'],
                   npu_data['global_representations'])

    print(f'\n{"=" * 60}')
    print('Done!')
    print('=' * 60)
