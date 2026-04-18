#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import triton
import triton.language as tl
import numpy as np
import torch
import sys
import os

# 添加 scripts 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import test_common


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@torch.inference_mode()
def triton_layer_norm(x, weight, bias, eps=1e-5):
    """Triton实现的层归一化"""
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y


def torch_layer_norm(x, weight, bias, eps=1e-5):
    """PyTorch参考实现的层归一化"""
    return torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias, eps=eps)


def calculate_errors(y_cal, y_ref):
    """计算误差指标"""
    # 转换为 CPU 张量进行计算
    y_cal_cpu = y_cal.cpu().numpy()
    y_ref_cpu = y_ref.cpu().numpy()
    
    # 计算绝对误差
    abs_error = np.abs(y_cal_cpu - y_ref_cpu)
    
    # 计算相对误差（避免除以零）
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs((y_cal_cpu - y_ref_cpu) / (y_ref_cpu + 1e-12))
    
    # 计算误差指标
    mere = np.mean(rel_error)
    mare = np.max(rel_error)
    max_abs_error = np.max(abs_error)
    
    return mere, mare, max_abs_error


def generate_report():
    """生成层归一化精度报告"""
    # 支持的数据类型
    supported_dtypes = ['float16', 'float32']
    
    # 测试参数
    test_shapes = [(128, 128), (256, 256), (512, 512)]
    eps = 1e-5
    
    # 生成报告文件
    report_path = os.path.join(os.path.dirname(__file__), 'layer_norm_precision_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("================================================================================\n")
        f.write("                              生态算子精度验证报告                               \n")
        f.write("                                层归一化 (Layer Norm)                               \n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("[验证配置]:\n")
        f.write(f"  算子名称: triton_layer_norm\n")
        f.write(f"  测试形状: {', '.join(str(shape) for shape in test_shapes)}\n")
        f.write(f"  数据类型: {', '.join(supported_dtypes)}\n")
        f.write(f"  Epsilon: {eps}\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("[精度标准]:\n")
        f.write("  float16: 相对误差 rtol=1e-03, atol=1e-03\n")
        f.write("  float32: 相对误差 rtol=1e-04, atol=1e-04\n")
        f.write("  bfloat16: 相对误差 rtol=1e-02, atol=1e-02\n")
        f.write("  int32/int64/int16/int8: 完全相等\n")
        f.write("  uint32/uint64/uint16/uint8: 完全相等\n")
        f.write("  bool: 完全相等\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("[验证结果]:\n")
        
        total_passed = 0
        total_tests = 0
        
        # 存储详细结果
        detailed_results = []
        
        for dtype in supported_dtypes:
            for shape in test_shapes:
                total_tests += 1
                try:
                    # 生成输入数据
                    np_x = test_common.generate_numpy(shape, dtype)
                    np_weight = test_common.generate_numpy((shape[-1],), dtype)
                    np_bias = test_common.generate_numpy((shape[-1],), dtype)
                    
                    # 转换为 PyTorch 张量并移动到 NPU
                    x = torch.from_numpy(np_x).to(eval('torch.' + dtype)).npu()
                    weight = torch.from_numpy(np_weight).to(eval('torch.' + dtype)).npu()
                    bias = torch.from_numpy(np_bias).to(eval('torch.' + dtype)).npu()
                    
                    # 计算参考结果
                    y_ref = torch_layer_norm(x, weight, bias, eps)
                    
                    # 计算Triton结果
                    y_cal = triton_layer_norm(x, weight, bias, eps)
                    
                    # 计算误差
                    mere, mare, max_abs_error = calculate_errors(y_cal, y_ref)
                    
                    # 验证精度
                    test_common.validate_cmp(dtype, y_cal, y_ref)
                    
                    detailed_results.append({
                        'dtype': dtype,
                        'shape': shape,
                        'status': 'PASS',
                        'mere': mere,
                        'mare': mare,
                        'max_abs_error': max_abs_error
                    })
                    total_passed += 1
                    
                except Exception as e:
                    detailed_results.append({
                        'dtype': dtype,
                        'shape': shape,
                        'status': 'FAIL',
                        'error': str(e)
                    })
        
        # 写入总体结果
        f.write(f"  测试总数: {total_tests}\n")
        f.write(f"  通过数量: {total_passed}\n")
        f.write(f"  失败数量: {total_tests - total_passed}\n")
        f.write(f"  总体结果: {'PASS' if total_passed == total_tests else 'FAIL'}\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("[详细误差指标]:\n")
        
        for result in detailed_results:
            f.write(f"  数据类型: {result['dtype']}, 形状: {result['shape']}\n")
            f.write(f"  测试结果: {result['status']}\n")
            
            if result['status'] == 'PASS':
                f.write(f"    平均相对误差(MERE): {result['mere']:.6e}\n")
                f.write(f"    最大相对误差(MARE): {result['mare']:.6e}\n")
                f.write(f"    最大绝对误差: {result['max_abs_error']:.6e}\n")
            else:
                f.write(f"    错误信息: {result['error'][:100]}...\n")
            f.write("\n")
        
        f.write("--------------------------------------------------------------------------------\n")
        f.write("[判定条件]:\n")
        f.write(f"  ✓ 所有数据类型测试通过: {'True' if total_passed == total_tests else 'False'}\n")
        f.write("================================================================================\n")
    
    print(f"层归一化精度报告已生成: {report_path}")
    
    # 打印报告内容
    with open(report_path, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    generate_report()
