---
name: external-gitcode-ascend-ascendc-operator-compile-debug
description: '编译安装 AscendC 算子并执行精度测试。TRIGGER when: 算子代码生成完成后需要编译验证、安装 whl 包、运行精度测试，或编译/测试失败需要排查。关键词：build.sh、编译、安装、whl、pytest、精度测试、编译错误、NPU
  测试。'
original-name: ascendc-operator-compile-debug
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# AscendC 算子编译安装与精度测试

编译 ascend-kernel 工程、安装 whl 包、生成并运行精度测试。通常由 `ascendc-operator-code-gen` skill 在代码生成完成后调用。

## 前置条件

- op_host、op_kernel 代码已生成
- ops.h、register.cpp、csrc/CMakeLists.txt 已更新（框架适配完成）
- CANN 环境可用

## 工作流程

### 阶段 0: 环境准备

**MANDATORY — 每次 shell 命令前必须加载环境。禁止硬编码路径，禁止自行搜索路径。**

#### 获取环境信息

若上游 skill（如 `ascendc-operator-dev`）已确认 CANN 路径和 Conda 环境名称，则直接使用，无需再次询问。

若未从上游获得环境信息，则按以下流程确认：

**CANN 环境**：
1. 检查环境变量 `ASCEND_HOME_PATH`（`echo $ASCEND_HOME_PATH`）
2. **若已设置**：直接使用，将其作为 `CANN_PATH`
3. **若未设置**：**MUST** 向用户询问 CANN 安装路径

**Conda 环境**：
1. 检查当前是否已激活 conda 环境（`echo $CONDA_DEFAULT_ENV`）
2. **若已激活**（值非 `base` 且非空）：直接使用当前环境
3. **若未激活或为 `base`**：**MUST** 向用户询问 conda 环境名称

#### 激活环境

确认后，在每次 shell 命令前加载：

```bash
source ${CANN_PATH}/*/set_env.sh
conda activate <env_name>
```

> **实战经验**：不加载环境会导致：(1) build.sh 找不到 AscendC 编译器；(2) pip/python 版本不对；(3) 运行测试时 torch_npu 找不到。建议在每条 shell 命令前都 source 一次。

### 阶段 1: 编译

```bash
cd ascend-kernel
chmod +x build.sh  # 模板复制后可能缺少执行权限
bash build.sh
```

**编译成功判断**:
```bash
ls ./output/ascend_kernel*.whl
```

存在 `.whl` 文件即为成功。

**编译失败**: 进入排错循环（见下方），最多 3 次。

### 阶段 2: 安装

```bash
cd ascend-kernel
pip install output/ascend_kernel*.whl --force-reinstall --no-deps
```

### 阶段 3: 生成测试文件

先检查 `ascend-kernel/tests/test_<op_name>.py` 是否已存在，已存在则跳过生成。

不存在时，在 `ascend-kernel/tests/` 下创建 `test_<op_name>.py`。

**测试文件结构**:

```python
#!/usr/bin/env python3
"""Unit tests for <op_name> NPU operator."""

import torch
import torch_npu
import pytest

try:
    import ascend_kernel
except ImportError:
    import os, glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    lib_pattern = os.path.join(project_root, "python/ascend_kernel/ascend_kernel/lib/*.so")
    lib_files = glob.glob(lib_pattern)
    if lib_files:
        torch.ops.load_library(lib_files[0])
    else:
        raise ImportError("Could not find ascend_kernel library")


def is_npu_available():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def device():
    if not is_npu_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


class Test<OpName>:

    @pytest.mark.parametrize("shape", [
        # 根据算子特点选择典型 shape
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_basic(self, device, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device=device)

        npu_result = torch.ops.npu.<op_name>(x)
        cpu_reference = <pytorch_reference>(x.cpu())

        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-5)
        assert torch.allclose(npu_result.cpu(), cpu_reference, rtol=rtol, atol=atol), \
            f"max diff = {(npu_result.cpu() - cpu_reference).abs().max().item()}"


def run_simple_test():
    """简单功能测试（无精度校验）"""
    if not is_npu_available():
        print("NPU not available, skipping test")
        return False

    device = torch.device("npu:0")
    try:
        for dtype in [torch.float16, torch.float32]:
            x = torch.randn(1024, dtype=dtype, device=device)
            result = torch.ops.npu.<op_name>(x)
            print(f"dtype={dtype}, output shape={result.shape} OK")
        print("Test PASSED!")
        return True
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_simple_test()
```

**测试用例设计要点**:

| 场景 | 参数化内容 | 精度标准 |
|------|-----------|---------|
| 基础 shape | 1D/2D/3D/4D | fp32: rtol=1e-5, fp16/bf16: rtol=1e-3 |
| 不同大小 | 1, 7, 256, 65536, 1M | 同上 |
| 边界值 | 全零、全负、极大值 | 绝对值比较 |
| 算子特有 | 根据设计文档中的约束条件 | 根据算子特性 |

**CPU 参考实现**: 使用 PyTorch 原生函数（如 `torch.acosh`、`torch.softmax` 等）。若无直接对应，手写 Python 参考逻辑。

**精度对比策略**:
- FP32 NPU 结果 vs FP32 CPU 结果：直接对比
- FP16 NPU 结果 vs CPU 参考：先在 CPU 上用 FP32 计算，再 `.to(dtype)` 转回后对比

**测试数据生成注意事项**:
- 检查算子的有效输入范围（如 acosh 要求 x >= 1，log 要求 x > 0，sqrt 要求 x >= 0）
- 生成数据时使用 `torch.rand() * scale + offset` 确保落入有效范围
- 包含边界值测试（如 acosh(1)=0, relu(0)=0）

### 阶段 4: 运行测试

**功能测试**（必须通过）:
```bash
cd ascend-kernel && python tests/test_<op_name>.py
```

**精度测试**（推荐执行）:
```bash
cd ascend-kernel && pytest tests/test_<op_name>.py -v
```

**测试通过标准**:
- 功能测试: 程序正常退出（exit code 0）
- 精度测试: 所有 pytest case 通过

## 排错循环（最多 3 次）

```
失败 → 分析错误日志 → 定位错误文件:行号 → 修复代码 → 重新编译/测试
  ↓ (仍失败)
第 2 次 → 重新分析，检查遗漏的 include/声明/CMakeLists
  ↓ (仍失败)
第 3 次 → 深入检查 API 用法、类型匹配、CMake 配置
  ↓ (仍失败)
停止 → 向用户报告详细错误信息和已尝试的修复
```

### 编译错误决策树

| 错误特征 | 修改位置 | 排查方向 |
|---------|---------|---------|
| `undeclared identifier` | `ops.h` 或 op_host cpp | 缺少函数声明或 include |
| `no matching function` | op_host cpp | 参数类型/顺序与 kernel 入口不匹配 |
| `undefined reference` / linker error | `csrc/CMakeLists.txt` | 源文件未加入 OP_SRCS 或 ascendc_library |
| `redefinition` | ops.h / register.cpp | 重复定义 |
| AscendC kernel 编译错误 | op_kernel cpp | API 使用方式或类型不支持 |

### 测试错误决策树

| 错误特征 | 排查方向 |
|---------|---------|
| `ImportError: ascend_kernel` | whl 未安装或 so 未生成 |
| `RuntimeError: ... not found` | register.cpp 注册名与调用名不一致 |
| `allclose failed` | 计算逻辑或精度问题，检查 kernel Compute 函数 |
| `shape mismatch` | op_host 输出 tensor shape 计算有误 |
| `SIGABRT / device error` | tiling 参数错误或 UB 越界 |

## 反模式清单

- **NEVER** 硬编码 CANN 路径或 Conda 环境路径，必须从上游获取或询问用户
- **NEVER** 自行搜索文件系统来猜测 CANN 或 Conda 路径
- **NEVER** 修改 `cmake/` 或 `csrc/utils/` 下的文件
- **NEVER** 在排错时删除其他算子的注册代码
- **NEVER** 跳过功能测试直接做精度测试
- **NEVER** 忽略编译 warning（尤其是类型截断相关）
- **NEVER** 在 EXEC_KERNEL_CMD 中传入右值（临时对象、字面量、表达式）
- **NEVER** 超过 3 次排错仍失败时继续尝试，应报告给用户

## 可修改文件范围

| 文件 | 修改内容 |
|------|---------|
| `csrc/ops/<op_name>/op_host/*.cpp` | host 端逻辑 |
| `csrc/ops/<op_name>/op_kernel/*.cpp` | kernel 端逻辑 |
| `csrc/ops.h` | 函数声明 |
| `csrc/register.cpp` | 算子注册 |
| `csrc/CMakeLists.txt` | 编译源文件列表 |
| `tests/test_<op_name>.py` | 测试文件 |
