---
name: external-gitcode-ascend-npu-adapter-reviewer
description: GPU代码到昇腾NPU适配审查专家。当用户需要将GPU上的代码（特别是深度学习、模型推理相关）迁移到华为昇腾NPU时，必须使用此skill进行全面审查。此skill能识别GPU到NPU迁移的堵点、编写适配脚本、生成验证方案，并输出完整的Markdown审查报告。触发场景包括：用户提到"NPU适配"、"昇腾迁移"、"GPU转NPU"、"Ascend"、"CANN"、"模型迁移"、"算子适配"等关键词，或者用户要求对GPU代码仓库进行审查并迁移到NPU平台。
original-name: npu-adapter-reviewer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-03-26'
synced-commit: 0a97c6e3999cf97425ca5ab07678e48089d79ff5
license: UNKNOWN
---


# NPU Adapter Reviewer - GPU到昇腾NPU适配审查专家

这是一个专门用于将GPU代码适配到华为昇腾NPU的Agent Skill。本技能覆盖完整的适配工作流：代码分析、堵点识别、适配脚本编写、验证方案设计、以及最终报告生成。

## 核心工作流程

### 阶段1：代码仓库获取与分析

**任务1.1：获取源代码**

根据用户提供的输入（本地路径或GitHub链接），获取完整的代码仓库：

```bash
# 如果是GitHub链接，先克隆
git clone <repo_url> /tmp/gpu_code_base
cd /tmp/gpu_code_base

# 如果是本地路径，直接分析
ls -la <local_path>
```

**任务1.2：全面代码扫描**

使用并行探索的方式分析代码结构：

1. **探索Agent 1 - 代码结构分析**
   - 找出所有Python文件、CUDA文件、C++文件
   - 识别项目目录结构
   - 找出主要的入口文件和配置文件

2. **探索Agent 2 - GPU依赖识别**
   - 搜索CUDA API调用（`cudaMalloc`, `cudaMemcpy`, `kernel<<<...>>>`, `torch.cuda`等）
   - 搜索PyTorch GPU相关代码（`.cuda()`, `.to('cuda')`, `torch.device('cuda')`等）
   - 搜索TensorRT相关代码
   - 搜索深度学习框架特定API（Transformer引擎、Flash Attention等）

3. **探索Agent 3 - 外部库依赖**
   - 搜索`import`和`from ... import`语句
   - 识别所有第三方库依赖
   - 检查是否有NPU不支持的库

**任务1.3：生成代码结构报告**

输出以下信息：
- 项目总文件数、代码行数
- 文件类型分布（Python/CUDA/C++/其他）
- 主要依赖库列表
- 核心模块及其功能描述

### 阶段2：GPU到NPU迁移堵点识别

**任务2.1：算子兼容性分析**

逐类识别GPU专用算子在NPU上的兼容性：

| 堵点类别 | GPU典型实现 | NPU替代方案 | 迁移难度 |
|---------|------------|------------|---------|
| CUDA核心算子 | `__global__`, `__device__`函数 | Ascend C算子 / ATB | 高 |
| 内存操作 | `cudaMallocHost`, `cudaMallocManaged` | `aclrtMalloc`, `HI_MPI_MALLOC` | 中 |
| 流和事件 | `cudaStream_t`, `cudaEvent_t` | `aclrtStream`, `aclrtEvent` | 中 |
| cuBLAS/cuDNN | `cublasGemmEx`, `cudnnConvolutionForward` | `aclblasGemmEx`, 算子融合 | 高 |
| Flash Attention | `flash_attn_varlen_func` | 昇腾Flash Attention算子 | 中 |
| 自定义算子 | PyTorch CUDA扩展 | ATC/ACL算子 | 高 |
| AMP/混合精度 | `torch.cuda.amp` | `ascend_mixed_precision` | 低 |

**任务2.2：识别具体堵点**

对每个GPU API调用，生成以下分析：

```
### 堵点编号: #001
- **文件位置**: `src/attention/cuda_impl.cu:142`
- **GPU API**: `cudaStreamCreate(&stream)`
- **NPU替代**: `aclrtCreateStream(&stream)`
- **迁移方案**: 
  1. 替换头文件 `aclrt.h`
  2. 替换API调用
  3. 处理错误码差异
- **预估工作量**: 0.5人天
- **影响范围**: 全局流管理
```

**任务2.3：生成堵点清单**

输出完整的堵点列表，按影响范围和迁移难度排序。

### 阶段3：适配脚本编写

**任务3.1：创建NPU适配层**

根据识别的堵点，创建适配脚本：

1. **创建 `npu_compat.py` - Python层兼容适配**
   ```python
   # 自动检测运行设备
   def get_device():
       if is_npu_available():
           return "npu"
       elif is_cuda_available():
           return "cuda"
       else:
           return "cpu"
   
   # 替换torch.cuda调用
   def to_device(tensor):
       device = get_device()
       if device == "npu":
           return tensor.npu()
       elif device == "cuda":
           return tensor.cuda()
       return tensor
   ```

2. **创建 `npu_ops.py` - NPU算子封装**
   - 将所有CUDA核心算子封装为NPU版本
   - 保留原有接口，内部实现NPU适配

3. **创建 `build_npu.sh` - 编译脚本**
   - ASCEND C算子编译命令
   - 依赖环境检查
   - 错误诊断

**任务3.2：修改原有代码**

生成修改后的代码文件，保留原文件并创建`.npu`版本：
- 替换所有GPU特定调用
- 添加设备检测逻辑
- 添加回退机制

### 阶段4：验证方案设计

**任务4.1：创建验证脚本**

根据适配内容，生成验证脚本 `verify_npu.sh`：

```bash
#!/bin/bash
# NPU适配验证脚本

echo "=== 1. 环境检查 ==="
check_npu_env() {
    # 检查NPU驱动
    ls -la /dev/*npu* 2>/dev/null || echo "Warning: NPU device not found"
    # 检查CANN
    echo $ASCEND_TOOLKIT_HOME
    # 检查Python包
    python3 -c "import torch; print('PyTorch version:', torch.__version__)"
    python3 -c "import torch_npu; print('torch_npu installed')"
}

echo "=== 2. 模块导入测试 ==="
test_imports() {
    cd <project_path>
    python3 -c "import npu_compat; print('npu_compat OK')"
    python3 -c "import npu_ops; print('npu_ops OK')"
}

echo "=== 3. 功能验证 ==="
test_functions() {
    # 运行基础测试
    python3 -m pytest tests/test_npu_*.py -v
    # 验证算子精度
    python3 scripts/verify_precision.py
}

echo "=== 4. 性能基准测试 ==="
benchmark() {
    python3 scripts/benchmark.py --device npu --compare cuda
}
```

**任务4.2：精度验证脚本**

生成 `verify_precision.py`：
```python
import numpy as np

def verify_npu_precision(cuda_result, npu_result, rtol=1e-3, atol=1e-3):
    """验证NPU与GPU输出精度差异"""
    diff = np.abs(cuda_result - npu_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    passed = np.allclose(cuda_result, npu_result, rtol=rtol, atol=atol)
    return {
        "passed": passed,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rtol": rtol,
        "atol": atol
    }
```

### 阶段5：审查报告生成

**任务5.1：生成Markdown报告**

根据验证结果，生成完整的审查报告：

```markdown
# GPU到昇腾NPU适配审查报告
# CodeReview_Results_YYYY-MM-DD.md

## 1. 执行摘要

| 项目 | 内容 |
|-----|------|
| 原始代码仓库 | `<repo_url>` 或 `<local_path>` |
| 审查日期 | YYYY-MM-DD |
| 适配状态 | ✅ 完全适配 / ⚠️ 部分适配 / ❌ 适配失败 |
| 识别堵点总数 | XX个 |
| 已适配堵点 | XX个 |
| 剩余堵点 | XX个 |

## 2. 原始代码分析

### 2.1 代码结构概览
- 总文件数：XX
- Python代码行数：XX
- CUDA/C++代码行数：XX
- 核心模块：...

### 2.2 依赖分析
| 库名 | 版本 | NPU兼容性 | 替代方案 |
|-----|------|----------|---------|
| torch | 2.x | ✅ 兼容 | torch_npu |
| flash-attn | 2.x | ⚠️ 部分 | 昇顿Flash Attention |

## 3. 迁移堵点详细分析

### 3.1 算子兼容性问题

#### 问题 #001: CUDA Stream管理
- **文件**: `src/utils/stream_manager.py:45`
- **GPU API**: `cudaStreamCreate`
- **问题描述**: 使用CUDA流管理异步执行
- **NPU替代**: `aclrtCreateStream`
- **影响范围**: 全局，影响所有异步操作
- **迁移建议**: 
  ```python
  # 修改前
  import torch.cuda
  stream = torch.cuda.Stream()
  
  # 修改后
  import torch_npu
  stream = torch.npu.Stream()
  ```
- **状态**: ✅ 已适配 / ⚠️ 待处理

#### 问题 #002: Flash Attention算子
- **文件**: `src/attention/flash_attn_impl.py:78`
- **GPU API**: `flash_attn_varlen_func`
- **问题描述**: 使用Flash Attention加速注意力计算
- **NPU替代**: Ascend flash_attn算子或MindSpore flash_attention
- **影响范围**: 高，核心推理性能
- **迁移建议**: 
  ```python
  # 修改前
  from flash_attn import flash_attn_func
  output = flash_attn_func(q, k, v)
  
  # 修改后
  # 方案1: 使用torch_npu的算子
  import torch_npu
  output = torch_npu.npu_flash_attention(q, k, v)
  
  # 方案2: 使用ATB库
  from ascend_toolkit import flash_attention
  output = flash_attention(q, k, v)
  ```
- **状态**: ✅ 已适配 / ⚠️ 待处理

### 3.2 模型加载与权重管理问题

#### 问题 #003: GPU权重格式
- **文件**: `src/model/loader.py:112`
- **问题描述**: 权重以CUDA格式存储，直接加载会失败
- **迁移建议**: 
  ```python
  # 修改前
  state_dict = torch.load(weights_path)
  model.load_state_dict(state_dict)
  
  # 修改后
  state_dict = torch.load(weights_path, map_location='cpu')
  # 转换权重
  for k, v in state_dict.items():
      if isinstance(v, torch.Tensor):
          state_dict[k] = v.npu()
  model.load_state_dict(state_dict)
  ```
- **状态**: ✅ 已适配

### 3.3 计算性能瓶颈

#### 问题 #004: 算子融合缺失
- **文件**: `src/model/inference.py:89`
- **问题描述**: 多个独立算子导致性能下降
- **迁移建议**: 使用ATC进行算子融合优化
- **预估性能提升**: 20-30%
- **状态**: ⚠️ 待处理

### 3.4 NPU内存与KV Cache管理

#### 问题 #005: 动态内存分配
- **文件**: `src/cache/kv_cache.py:56`
- **问题描述**: 使用CUDA动态内存分配
- **迁移建议**: 使用固定内存池
- **状态**: ⚠️ 待处理

### 3.5 Python-C++边界问题

#### 问题 #006: C++扩展编译
- **文件**: `src/utils/gpu_ext.cpp:145`
- **问题描述**: CUDA C++扩展需要重新编译
- **迁移建议**: 使用Ascend C重写或使用ATB
- **状态**: ⚠️ 待处理

### 3.6 并发与异步问题

#### 问题 #007: 多流并发
- **文件**: `src/server/request_handler.py:78`
- **问题描述**: 使用CUDA流实现并发
- **迁移建议**: 重构为进程级并发
- **状态**: ⚠️ 待处理

### 3.7 配置与可维护性问题

#### 问题 #008: 硬编码设备
- **文件**: `src/config.py:23`
- **问题描述**: 配置中硬编码`cuda:0`
- **迁移建议**: 改为设备检测
- **状态**: ✅ 已适配

## 4. 适配代码清单

### 4.1 新增文件

| 文件名 | 功能 | 状态 |
|-------|------|------|
| `npu_compat.py` | 设备检测与兼容层 | ✅ |
| `npu_ops.py` | NPU算子封装 | ✅ |
| `build_npu.sh` | 编译脚本 | ✅ |
| `verify_npu.sh` | 验证脚本 | ✅ |

### 4.2 修改文件

| 文件名 | 修改内容 | 状态 |
|-------|---------|------|
| `src/attention/flash_attn.py` | 替换为NPU算子 | ✅ |
| `src/model/loader.py` | 添加权重转换 | ✅ |
| `src/utils/stream_manager.py` | Stream适配 | ✅ |

## 5. 验证结果

### 5.1 环境验证
- [x] NPU驱动已安装
- [x] CANN Toolkit已配置
- [x] torch_npu已安装
- [x] Python模块可导入

### 5.2 功能验证
- [x] 基础模块导入测试通过
- [x] 设备检测功能正常
- [x] 前向推理执行成功
- [x] 权重加载转换正常

### 5.3 精度验证
- [x] 推理结果与GPU差异 < 0.1%
- [ ] 性能测试待执行（需要NPU硬件）

### 5.4 问题汇总
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 已解决 | XX | - |
| 待解决 | XX | 高/中/低 |

## 6. 适配指南

### 6.1 前置条件

```bash
# 1. 安装CANN Toolkit
# 下载地址: https://www.hiascend.com/software/aiengine

# 2. 安装torch_npu
pip install torch torch_npu

# 3. 验证安装
python3 -c "import torch; import torch_npu; print('NPU available:', torch_npu.is_npu_available())"
```

### 6.2 快速适配步骤

**步骤1: 克隆并进入项目**
```bash
git clone <repo_url>
cd <project_name>
```

**步骤2: 安装依赖**
```bash
pip install -r requirements-npu.txt
```

**步骤3: 运行验证**
```bash
bash verify_npu.sh
```

**步骤4: 执行推理**
```bash
python3 run_npu.py --model <model_path> --input <input_data>
```

### 6.3 常见问题排查

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 导入失败 | CANN未正确安装 | 重新配置环境变量 |
| 算子不支持 | NPU不支持该算子 | 使用ATB替代或自研算子 |
| 内存溢出 | 批处理过大 | 减小batch_size |
| 精度不达标 | 混合精度配置问题 | 检查AMP配置 |

## 7. 后续工作建议

### 7.1 短期（1周内）
- [ ] 完成剩余堵点的适配
- [ ] 在真实NPU硬件上进行性能测试
- [ ] 优化算子融合

### 7.2 中期（1个月内）
- [ ] 完善错误处理机制
- [ ] 添加日志和监控
- [ ] 性能调优

### 7.3 长期
- [ ] 持续跟进CANN更新
- [ ] 自动化测试流程
- [ ] 文档完善

---

**报告生成时间**: YYYY-MM-DD HH:mm:ss
**适配工程师**: AI Agent (NPU Adapter Reviewer)
**报告版本**: v1.0
```

**任务5.2：输出报告**

将报告保存到当前目录：
```
CodeReview_Results_YYYY-MM-DD.md
```

## 输出要求

1. **报告格式**: 必须是Markdown格式
2. **文件命名**: `CodeReview_Results_运行当天的日期.md`（格式：YYYY-MM-DD）
3. **保存位置**: 当前工作目录
4. **内容完整性**: 必须包含上述所有章节

## 特殊处理规则

### 如果验证完全通过
- 输出"适配成功"的状态
- 提供完整的适配指南
- 包含端到端运行说明

### 如果验证未完全通过
- 详细说明每个失败项
- 提供具体的修复建议
- 给出修改后的代码
- 标注需要人工介入的部分

## 知识参考

在执行过程中，可参考以下资料（按需加载）：
- `references/ascend_npu_best_practices.md` - 昇腾NPU最佳实践
- `references/cann_migration_guide.md` - CANN迁移指南
- `references/npu_python_api.md` - NPU Python API参考

请使用此skill完成GPU到昇腾NPU的完整适配审查工作。
