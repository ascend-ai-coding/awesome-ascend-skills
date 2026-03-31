#!/usr/bin/env python3
"""
Triton算子精度验证工具

用于验证迁移后的Triton算子精度是否正确。
"""

import torch
import torch_npu


def verify_accuracy(result: torch.Tensor, reference: torch.Tensor, 
                    dtype: torch.dtype, name: str = "test") -> bool:
    """
    验证结果精度
    
    Args:
        result: 待验证的结果
        reference: 参考结果
        dtype: 数据类型
        name: 测试名称
    
    Returns:
        bool: 是否通过验证
    """
    print(f"\n{'='*60}")
    print(f"验证: {name}")
    print(f"{'='*60}")
    
    # 检查形状
    if result.shape != reference.shape:
        print(f"❌ 形状不匹配: result {result.shape} vs reference {reference.shape}")
        return False
    print(f"✅ 形状匹配: {result.shape}")
    
    # 检查NaN
    has_nan_result = torch.isnan(result).any().item()
    has_nan_ref = torch.isnan(reference).any().item()
    if has_nan_result:
        print(f"❌ 结果包含NaN")
        return False
    print(f"✅ 无NaN")
    
    # 检查Inf
    has_inf_result = torch.isinf(result).any().item()
    has_inf_ref = torch.isinf(reference).any().item()
    if has_inf_result and not has_inf_ref:
        print(f"❌ 结果包含意外的Inf")
        return False
    print(f"✅ 无意外Inf")
    
    # 设置容差
    if dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 1e-3, 1e-3
    elif dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 0, 0
    
    # 精度对比
    try:
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        print(f"✅ 精度验证通过 (rtol={rtol}, atol={atol})")
        return True
    except AssertionError as e:
        print(f"❌ 精度验证失败: {e}")
        
        # 打印统计信息
        diff = (result - reference).abs()
        print(f"\n差异统计:")
        print(f"  最大差异: {diff.max().item()}")
        print(f"  平均差异: {diff.mean().item()}")
        print(f"  差异标准差: {diff.std().item()}")
        
        return False


def run_test_suite(test_func, device='npu'):
    """
    运行测试套件
    
    Args:
        test_func: 测试函数，接受device参数
        device: 设备类型
    """
    print(f"\n{'='*60}")
    print(f"Triton算子精度验证")
    print(f"设备: {device}")
    print(f"{'='*60}")
    
    if device == 'npu' and not torch.npu.is_available():
        print("❌ NPU设备不可用")
        return False
    
    try:
        test_func(device)
        print(f"\n{'='*60}")
        print(f"✅ 所有测试通过")
        print(f"{'='*60}")
        return True
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 测试失败: {e}")
        print(f"{'='*60}")
        return False


def diagnose_result(result: torch.Tensor, reference: torch.Tensor):
    """
    诊断结果问题
    
    Args:
        result: 待诊断的结果
        reference: 参考结果
    """
    print(f"\n{'='*60}")
    print(f"结果诊断")
    print(f"{'='*60}")
    
    # 基本信息
    print(f"结果形状: {result.shape}")
    print(f"结果类型: {result.dtype}")
    print(f"结果设备: {result.device}")
    
    # 统计信息
    print(f"\n结果统计:")
    print(f"  最小值: {result.min().item()}")
    print(f"  最大值: {result.max().item()}")
    print(f"  均值: {result.mean().item()}")
    print(f"  标准差: {result.std().item()}")
    
    # NaN/Inf检查
    nan_count = torch.isnan(result).sum().item()
    inf_count = torch.isinf(result).sum().item()
    print(f"\n异常值:")
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    
    # 与参考对比
    if reference is not None:
        diff = (result - reference).abs()
        print(f"\n与参考对比:")
        print(f"  最大差异: {diff.max().item()}")
        print(f"  平均差异: {diff.mean().item()}")
        print(f"  差异>0.01的数量: {(diff > 0.01).sum().item()}")
        print(f"  差异>0.1的数量: {(diff > 0.1).sum().item()}")


if __name__ == "__main__":
    # 示例用法
    print("Triton算子精度验证工具")
    print("使用方法:")
    print("  from verify_accuracy import verify_accuracy, run_test_suite")
    print("  ")
    print("  def test_my_kernel(device):")
    print("      x = torch.randn(100, 100, device=device)")
    print("      result = my_kernel(x)")
    print("      reference = torch_reference(x)")
    print("      verify_accuracy(result, reference, x.dtype, 'my_test')")
    print("  ")
    print("  run_test_suite(test_my_kernel)")
