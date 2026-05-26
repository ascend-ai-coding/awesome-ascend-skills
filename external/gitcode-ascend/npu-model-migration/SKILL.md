---
name: external-gitcode-ascend-npu-model-migration
description: '自动化将 PyTorch 模型迁移到华为昇腾 NPU。Use when: 用户请求将模型迁移到 NPU、适配 NPU、在 NPU 上跑通模型、迁移到昇腾。'
original-name: npu-model-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# NPU Model Migration Skill

将任意 PyTorch 模型迁移到华为昇腾 NPU 并验证运行。

## 触发条件

**使用此 skill 当用户说：**
- "把这个模型迁移到 NPU"
- "帮我适配到昇腾"
- "在 NPU 上跑通这个模型"
- "迁移到 Ascend"
- "帮我看看这个模型能不能在 NPU 上跑"

## 核心理念

**模型迁移是一项需要"诊断能力"的任务，而非简单的"查找替换"。**

大多数 PyTorch 代码天生支持 NPU，只需要解决：
1. 设备检测逻辑
2. 第三方库兼容性
3. 不支持的 API

每个模型遇到的问题都不同，需要**分析问题→定位根因→修复→验证**的迭代循环。

---

## 迁移流程（七阶段）

```
┌─────────────────────────────────────────────┐
│ 阶段 1: 目标分析 (必做)                        │
│   1.1 环境分析 → 依赖、框架、Python 版本        │
│   1.2 代码分析 → 测试入口、训练/推理流程        │
│   1.3 迁移难度评估 → 预估需要改动的范围         │
└─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────┐
│ 阶段 1.5: 快速尝试 (强烈推荐!)                 │
│   1.5.1 尝试 transfer_to_npu                  │
│   1.5.2 检查是否还有遗漏项                     │
│   1.5.3 验证能否直接跑通                       │
└─────────────────────────────────────────────┘
                        ↓ (若失败则进入阶段2)
┌─────────────────────────────────────────────┐
│ 阶段 2: 方案设计 (必做)                        │
│   2.1 识别需要修改的文件                       │
│   2.2 确定迁移策略                            │
│   2.3 向用户反馈计划，等待确认                 │
└─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────┐
│ 阶段 3: 代码迁移                              │
│   3.1 设备适配                               │
│   3.2 依赖修复                               │
│   3.3 API 替换                               │
└─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────┐
│ 阶段 4: NPU 验证 (必做)                       │
│   4.1 克隆代码到 NPU 服务器                    │
│   4.2 安装依赖、运行测试                       │
│   4.3 收集输出、验证结果                       │
└─────────────────────────────────────────────┘
                        ↓
          ┌─────────────────────────────────┐
          │ 阶段 5: 调试与迭代 (循环)         │
          │   5.1 分析报错                    │
          │   5.2 定位根因                    │
          │   5.3 修复代码                    │
          │   5.4 重新验证                    │
          └─────────────────────────────────┘
                        ↓
                  (循环直到通过)
                        ↓
┌─────────────────────────────────────────────┐
│ 阶段 6: 输出迁移报告 (必做)                    │
│   6.1 生成报告 Markdown                       │
│   6.2 记录修改内容与验证结果                   │
│   6.3 归档到 references/cases/               │
└─────────────────────────────────────────────┘
```

---

## 阶段 1: 目标分析

### 1.1 环境分析

从项目入口获取关键信息：

```bash
# 查找依赖文件
find . -name "requirements*.txt" -o -name "setup.py" -o -name "pyproject.toml" | head -5

# 查找 README
find . -name "README*.md" -o -name "readme*.md" | head -3
```

**需要获取：**
- Python 版本要求
- PyTorch 版本
- 关键依赖（numpy、scikit-learn、transformers 等）
- 安装方式

### 1.2 代码分析

**找到测试/运行入口：**
```bash
# 查找测试目录
find . -type d -name "test*" -o -type d -name "example*" | head -5

# 查找训练/推理脚本
find . -name "train*.py" -o -name "main*.py" -o -name "run*.py" | head -10
```

**分析代码结构：**
- 训练流程：`compile → fit → evaluate` 还是自定义循环
- 测试框架：pytest / unittest / 手动脚本
- 设备选择逻辑：`cuda.is_available()` 的位置
- 数据加载方式：DataLoader、batch 处理

### 1.3 README 分析（重点！）

**README 是迁移工作的重要参考依据，必须仔细阅读。**

README 通常包含以下关键信息：

| 内容 | 重要性 | 迁移时的作用 |
|------|--------|--------------|
| **环境配置** | ⭐⭐⭐ | 确定 Python、PyTorch、依赖版本 |
| **数据集准备** | ⭐⭐⭐ | 了解数据格式、下载方式、存放路径 |
| **训练命令** | ⭐⭐⭐ | 确定训练入口、参数配置、如何验证 |
| **测试命令** | ⭐⭐⭐ | 确定测试方式、评估指标 |
| **预训练模型** | ⭐⭐ | 是否需要下载预训练权重 |
| **结果复现** | ⭐⭐ | 验证迁移是否成功的参考标准 |

**分析要点：**

```bash
# 查找 README 文件
find . -name "README*.md" -o -name "readme*.md" -o -name "INSTALL.md" | head -5
```

1. **环境依赖**
   - 查看 `requirements.txt` 或 `setup.py`
   - 确认 Python、PyTorch 版本要求
   - 记录所有第三方库

2. **数据准备**
   - 数据集名称、下载链接
   - 数据格式（CSV、JSON、TFRecord 等）
   - 数据目录结构
   - 预处理方式

3. **训练流程**
   - 训练命令示例
   - 关键超参数
   - 训练脚本入口
   - 需要修改的配置

4. **验证方式**
   - 如何测试模型
   - 评估指标（accuracy、loss、AUC 等）
   - 预期的输出格式
   - **这是迁移后验证成功的重要参考！**

**迁移后的验证应参考 README 中的测试/验证方式，确保输出格式和指标与原版一致。**

### 1.4 迁移难度评估

| 复杂度 | 特征 | 预估改动 |
|--------|------|----------|
| **简单** | 纯 PyTorch 模型，无第三方库依赖 | 少量设备适配 |
| **中等** | 有 sklearn、numpy 等依赖 | 设备适配 + 依赖修复 |
| **复杂** | 有自定义 CUDA 算子、多卡分布式 | 需要额外适配 |

---

## 阶段 1.5: 快速尝试（强烈推荐）

**在完成目标分析后，务必先尝试这个"偷懒"办法！**

### 1.5.1 什么是 transfer_to_npu

`torch_npu.contrib.transfer_to_npu` 是一个**自动重定向工具**，通过 monkey-patch 将代码中的 `torch.cuda.*` API 自动劫持到 `torch.npu.*`。

**它能自动处理：**
- `torch.cuda.is_available()` → `torch.npu.is_available()`
- `tensor.cuda()` → `tensor.npu()`
- `torch.cuda.empty_cache()` → `torch.npu.empty_cache()`
- `torch.cuda.set_device(x)` → `torch.npu.set_device(x)`
- `device='cuda:0'` → `device='npu:0'`
- `backend="nccl"` → `backend="hccl"` (分布式)
- `torch.profiler` CUDA 相关 → NPU 版本
- transformers/trl/peft/accelerate 等主流库的部分 API

### 1.5.2 快速尝试方法

**在原代码入口文件头部添加：**

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

然后直接运行测试。如果成功，**无需任何其他修改**！

```bash
# 使用 NPU 卡 0,1（根据实际情况修改）
ASCEND_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 python run.py
```

### 1.5.3 检查遗漏项

如果 `transfer_to_npu` 不能完全跑通，需要手动检查以下遗漏项：

| 遗漏项 | 搜索模式 | 修复方法 |
|--------|----------|----------|
| autocast 设备字符串 | `autocast.*cuda` | `'cuda'` → `'npu'` |
| 设备类型判断 | `device.*cuda\|is_cuda` | `cuda` → `npu` |
| 第三方库 CUDA 调用 | 库特定 | 需要手动修复 |

```bash
# 搜索遗漏项
grep -rn "autocast.*cuda\|'cuda'\|is_cuda\|\.cuda(" --include="*.py" .
```

### 1.5.4 验证结果判断

**成功标准：**
- 训练正常输出：`Epoch 1/10, Loss: 0.123`
- 推理正常输出：`Output shape: torch.Size([...])`
- 评估正常输出：`AUC: 0.755`

**失败处理：**
- 如果遇到报错，记录错误信息
- 进入阶段 2 进行深度迁移

---

## 阶段 2: 方案设计（深度迁移用）

> ⚠️ 只有在阶段 1.5（快速尝试）失败后才进入此阶段

### 2.1 识别需要修改的文件

基于代码分析，找出需要修改的文件：

1. **设备相关** - 包含 `cuda.is_available()`、`.cuda()` 的文件
2. **入口文件** - main.py、train.py、run.py
3. **配置相关** - config.yaml、args.py

### 2.2 确定迁移策略

```
策略 A: 最小改动（推荐）
├── 保留原代码不变
├── 创建 npu 版本目录
└── 只修改必要的设备相关代码

策略 B: 条件适配
├── 添加 NPU 检测逻辑
├── 同时支持 GPU/NPU
└── 通过环境变量切换
```

### 2.3 向用户反馈 (必做)

**在开始修改之前，必须向用户确认：**

```markdown
## 迁移计划

### 目标模型
- 模型名称: xxx
- 仓库: xxx

### 分析结果
- 依赖: torch >= 1.x, sklearn, ...
- 测试方式: pytest tests/xxx.py
- 预估复杂度: 中等

### 快速尝试
- 已尝试 transfer_to_npu: ❌ 失败
- 失败原因: xxx

### 迁移策略
- 策略 A: 最小改动
- 需要修改的文件:
  1. xxx.py (设备适配)
  2. xxx.py (依赖修复)

### 下一步
- 等待确认后开始修改
- 修改后会在 NPU 服务器上验证
```

---

## 阶段 3: 代码迁移（深度迁移用）

### 3.1 设备适配 (核心)

**原则：尽量少改，只改必要的设备相关代码。**

#### 3.1.1 基础设备检测

```python
# 方案 1: 环境变量控制
import os
CPU_ENABLE = os.environ.get("MODEL_CPU_ONLY", "false").upper() == "TRUE"
GPU_ENABLE = torch.cuda.is_available() and not CPU_ENABLE
NPU_ENABLE = not GPU_ENABLE and not CPU_ENABLE

if NPU_ENABLE:
    import torch_npu
```

#### 3.1.2 设备替换规则

| 原代码 | 替换为 | 说明 |
|--------|--------|------|
| `torch.cuda.is_available()` | `torch_npu.npu.is_available()` | 设备检测 |
| `torch.cuda.current_device()` | `torch_npu.npu.current_device()` | 当前设备 |
| `torch.cuda.set_device(x)` | `torch_npu.npu.set_device(x)` | 设置设备 |
| `.cuda()` | `.npu()` | 张量/模型移动 |
| `"cuda"` | `"npu"` | 设备字符串 |
| `torch.cuda.amp` | `torch_npu.npu.amp` | 混合精度 |
| `backend="nccl"` | `backend="hccl"` | 分布式训练 |

#### 3.1.3 设备初始化模板

```python
def get_device(prefer_npu=True):
    """自动获取可用设备"""
    import os
    import torch

    # 优先级: 环境变量 > NPU > GPU > CPU
    if os.environ.get("MODEL_CPU_ONLY", "").upper() == "TRUE":
        return torch.device("cpu")

    if prefer_npu:
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return torch.device("npu:0")
        except ImportError:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")
```

### 3.2 依赖修复

**常见第三方库问题：**

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| sklearn 版本不兼容 | NPU 环境 sklearn 版本较旧 | 升级或降级版本 |
| numpy 兼容 | 某些 numpy 函数 NPU 不支持 | 用 torch 替代 |
| pandas 兼容 | 某些操作在 NPU 上不支持 | 转回 CPU 计算 |

**处理原则：**
- 非核心依赖尽量不动
- 核心依赖问题优先尝试升级/降级版本
- 必要时将部分计算放回 CPU

### 3.3 API 替换

**常见需要替换的 API：**

```python
# 需要替换
torch.cuda.get_device_capability()  # NPU 无此概念
torch.cuda.comm.scatter/gather     # NPU 不支持

# 可以保留（自动兼容）
torch.nn.Module.to("cuda")         # 内部会处理
torch.tensor(device="cuda")        # 内部会处理
```

**详细 API 支持列表见：** `references/npu-api-mapping.md`

---

## 阶段 4: NPU 验证（通用）

### 4.1 环境准备

```bash
# 设置 NPU 卡号（根据环境约束）
export ASCEND_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

### 4.2 运行测试

**直接在本地 NPU 环境执行：**

```bash
# 1. 克隆代码（如果不在本地）
git clone <repo-url> -b <branch-name>
cd <project-dir>

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行测试（使用指定的 NPU 卡）
ASCEND_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 python run.py --config config.yaml
# 或
ASCEND_VISIBLE_DEVICES=0,1 ASCEND_RT_VISIBLE_DEVICES=0,1 pytest tests/test_model.py -v
```

### 4.3 验证标准

**必须看到以下输出之一才算成功：**
- 训练过程：`Epoch 1/10, Loss: 0.123`
- 推理过程：`Output shape: torch.Size([...])`
- 评估指标：`AUC: 0.755`

**禁止：**
- 只看到 "import 成功" 就说通过
- 只看到 "开始训练" 就说通过
- 没有具体数值输出

---

## 阶段 5: 调试与迭代（通用）

### 5.1 报错分类

| 错误类型 | 典型特征 | 排查方向 |
|----------|----------|----------|
| `ModuleNotFoundError` | 缺少模块 | 安装依赖、检查版本 |
| `RuntimeError: NPU not found` | 设备问题 | 检查 ASCEND_VISIBLE_DEVICES |
| `NotImplementedError` | API 不支持 | 查文档，找替代方案 |
| `TypeError` | 类型不匹配 | 检查 dtype、device |
| `AssertionError` | 精度/值校验失败 | 调整 rtol/atol |
| `OOM` | 内存不足 | 减小 batch_size |

### 5.2 迭代流程

```
1. 分析错误信息
2. 判断错误类型
3. 搜索解决方案 (文档/案例)
4. 修改代码
5. 重新运行验证
6. 通过则结束，失败则回到步骤 1

最大迭代次数: 5 次
超过次数应向用户报告困难
```

### 5.3 常见问题速查

**Q: 提示 torch_npu 找不到**
A: 确认容器中 torch 和 torch_npu 版本匹配

**Q: 提示 NPU not found**
A: 检查 ASCEND_VISIBLE_DEVICES 环境变量是否正确设置

**Q: 提示 API 不支持**
A: 查看 references/npu-api-mapping.md 找替代 API

**Q: 精度不达标**
A: 检查 HF32 配置，尝试调整 rtol/atol

**Q: OOM 内存不足**
A: 减小 batch_size，使用 gradient checkpointing

---

## 阶段 6: 输出迁移报告（必做）

> ⚠️ 每个模型迁移完成后，必须生成迁移报告并归档

### 6.1 报告模板

使用标准模板：`references/migration-report-template.md`

**必须包含的内容：**
1. 基本信息（模型名称、类型、仓库、迁移日期、状态）
2. 环境信息（NPU 设备、Python/PyTorch/torch_npu 版本）
3. 迁移过程（快速尝试结果、修改的文件、依赖修复）
4. 验证结果（运行命令、输出日志、性能对比）
5. 遇到的问题与解决
6. 待优化项

### 6.2 输出位置

```
references/cases/<模型名小写>.md
```

例如：
- `references/cases/autoint.md`
- `references/cases/yolov5.md`
- `references/cases/deepfm.md`

### 6.3 报告示例

```markdown
# AutoInt NPU 迁移报告

## 基本信息

| 字段 | 内容 |
|------|------|
| **模型名称** | AutoInt |
| **模型类型** | 推荐系统（CTR 预估） |
| **仓库地址** | https://github.com/FuxiCTR/AutoInt |
| **迁移日期** | 2026-04-03 |
| **迁移状态** | ✅ 成功 |

## 环境信息

- NPU: Ascend 910B4 (30GB)
- Python: 3.9
- PyTorch: 2.3.0
- torch_npu: 2.6.0

## 迁移过程

### 1. 快速尝试
- transfer_to_npu: ❌ 失败
- 失败原因: sklearn 兼容性

### 2. 深度迁移
- 修改 fuxictr/metrics.py (sklearn 兼容性)
- 修改 fuxictr/pytorch/torch_utils.py (NPU 设备支持)

## 验证结果

运行命令:
```bash
ASCEND_VISIBLE_DEVICES=0 python run.py --config config.yaml
```

输出:
```
AUC: 0.755208
```

## 总结

迁移结论: 简单
```

### 6.4 注意事项

- **必须归档**：不输出报告 = 迁移未完成
- **验证结果**：必须包含实际运行输出（数值）
- **问题记录**：如实记录遇到的问题和解决方案
- **后续复用**：报告是 future migration 的重要参考

---

## 案例参考

### 案例结构

每个案例应该包含：
1. **模型信息** - 名称、仓库、特点
2. **遇到的问题** - 报错信息、分析过程
3. **解决方案** - 具体修改了什么
4. **验证结果** - 运行输出、指标

### 已验证案例

| 模型 | 复杂度 | 状态 | 关键修改 |
|------|--------|------|----------|
| AutoInt | 简单 | ✅ | sklearn 兼容性、设备检测 |
| DeepFM | 中等 | ✅ | 设备适配、HF32 |
| DIN | 中等 | ✅ | 分布式配置 |
| Wide&Deep | 简单 | ✅ | 设备检测 |

详细案例见：`references/cases/`

---

## 多模型项目处理

如果用户给的是一个**模型框架**（包含多个模型），如：
- TorchEasyRec（DeepFM、DIN、Wide&Deep...）
- FuxiCTR（AutoInt、DeepFM...）

**处理方式：**
1. 先告知用户这是一个框架
2. 询问用户要迁移哪个具体模型
3. 确定后按单模型流程处理

---

## 约束条件

1. **工作目录限制**：根据实际环境配置
2. **NPU 环境**：确保 NPU 驱动和 torch_npu 已正确安装
3. **NPU 卡号**：根据实际可用卡号设置，建议通过环境变量配置
4. **不修改原代码**：建议在并行目录进行修改，或使用 Git 分支
5. **必须验证**：没有在 NPU 上运行过，不说"迁移成功"

---

## 文件结构

```
npu-model-migration/
├── SKILL.md                         # 本文件
└── references/
    ├── README.md                    # 快速参考指南
    ├── npu-api-mapping.md           # API 映射表
    ├── common-issues.md             # 常见问题汇总
    ├── migration-report-template.md # 迁移报告模板
    └── cases/                       # 迁移案例
        └── autoint.md
```

---

## 参考资料

- [Ascend PyTorch 官方文档](https://gitcode.com/Ascend/pytorch)
- [RecSDK Benchmark](https://gitcode.com/Ascend/RecSDK)
- [torch_npu API 映射表](references/npu-api-mapping.md)

---

*Skill 版本: 1.2.0* - 新增迁移报告输出流程（阶段 6）
*更新日期: 2026-04-15*