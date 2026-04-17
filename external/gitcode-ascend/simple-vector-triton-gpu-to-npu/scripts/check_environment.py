#!/usr/bin/env python3
"""
NPU环境检查工具

用于检查NPU环境是否正确配置。
"""

import sys


def check_npu_environment():
    """检查NPU环境"""
    print("=" * 60)
    print("NPU环境检查")
    print("=" * 60)
    
    checks = []
    
    # 1. 检查torch_npu
    print("\n[1/5] 检查 torch_npu...")
    try:
        import torch_npu
        print("  ✅ torch_npu 已安装")
        checks.append(True)
    except ImportError:
        print("  ❌ torch_npu 未安装")
        print("     解决方案: pip install torch-npu")
        checks.append(False)
    
    # 2. 检查triton-ascend
    print("\n[2/5] 检查 triton-ascend...")
    try:
        import triton
        # 检查是否是triton-ascend版本
        if hasattr(triton, '__version__'):
            print(f"  ✅ triton 已安装 (版本: {triton.__version__})")
        else:
            print("  ✅ triton 已安装")
        checks.append(True)
    except ImportError:
        print("  ❌ triton 未安装")
        print("     解决方案: pip install triton-ascend")
        checks.append(False)
    
    # 3. 检查NPU设备
    print("\n[3/5] 检查 NPU 设备...")
    try:
        import torch
        import torch_npu
        
        if torch.npu.is_available():
            device_count = torch.npu.device_count()
            device_name = torch.npu.get_device_name(0)
            print(f"  ✅ NPU 设备可用")
            print(f"     设备数量: {device_count}")
            print(f"     设备名称: {device_name}")
            checks.append(True)
        else:
            print("  ❌ NPU 设备不可用")
            print("     解决方案: 检查驱动安装和环境变量")
            checks.append(False)
    except Exception as e:
        print(f"  ❌ NPU 检查失败: {e}")
        checks.append(False)
    
    # 4. 检查环境变量
    print("\n[4/5] 检查环境变量...")
    import os
    
    env_checks = []
    important_vars = [
        'ASCEND_HOME',
        'ASCEND_TOOLKIT_HOME',
        'LD_LIBRARY_PATH',
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {value[:50]}...")
            env_checks.append(True)
        else:
            print(f"  ⚠️  {var}: 未设置")
            env_checks.append(False)
    
    checks.append(any(env_checks))
    
    # 5. 运行简单测试
    print("\n[5/5] 运行简单测试...")
    try:
        import torch
        import torch_npu
        
        x = torch.randn(10, 10, device='npu')
        y = x * 2
        if y.device.type == 'npu':
            print("  ✅ 简单张量运算成功")
            checks.append(True)
        else:
            print("  ❌ 张量不在NPU上")
            checks.append(False)
    except Exception as e:
        print(f"  ❌ 简单测试失败: {e}")
        checks.append(False)
    
    # 总结
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ 环境检查通过 ({passed}/{total})")
        return True
    else:
        print(f"⚠️  环境检查未完全通过 ({passed}/{total})")
        print("\n建议:")
        if not checks[0]:
            print("  - 安装 torch-npu: pip install torch-npu")
        if not checks[1]:
            print("  - 安装 triton-ascend: pip install triton-ascend")
        if not checks[2]:
            print("  - 检查NPU驱动和环境配置")
        if not checks[4]:
            print("  - 检查NPU运行时环境")
        return False


def get_npu_info():
    """获取NPU设备信息"""
    try:
        import torch
        import torch_npu
        
        if not torch.npu.is_available():
            print("NPU设备不可用")
            return None
        
        info = {
            'device_count': torch.npu.device_count(),
            'current_device': torch.npu.current_device(),
            'device_name': torch.npu.get_device_name(0),
            'capability': torch.npu.get_device_capability(0),
        }
        
        # 尝试获取更多信息
        try:
            import triton.runtime.driver as driver
            device = torch.npu.current_device()
            properties = driver.active.utils.get_device_properties(device)
            info['num_aicore'] = properties.get('num_aicore', 'N/A')
            info['num_vectorcore'] = properties.get('num_vectorcore', 'N/A')
        except:
            pass
        
        return info
    except Exception as e:
        print(f"获取NPU信息失败: {e}")
        return None


if __name__ == "__main__":
    # 检查环境
    env_ok = check_npu_environment()
    
    # 获取设备信息
    if env_ok:
        print("\n" + "=" * 60)
        print("NPU设备信息")
        print("=" * 60)
        info = get_npu_info()
        if info:
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    sys.exit(0 if env_ok else 1)
