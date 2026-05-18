# AutoInt 迁移案例

## 模型信息

| 项目 | 内容 |
|------|------|
| 模型名称 | AutoInt |
| 仓库 | FuxiCTR/AutoInt |
| 类型 | 点击率预估 (CTR) |
| 复杂度 | 简单 |

## 问题背景

AutoInt 是 FuxiCTR 库中的一个模型，用于点击率预估。用户希望将其迁移到 NPU 上运行。

## 迁移过程

### 阶段 1: 目标分析

**环境分析：**
- 依赖：PyTorch, numpy, scikit-learn
- Python 版本：3.8+
- 安装方式：`pip install -e .`

**代码分析：**
- 入口：`train.py` 或项目目录下的训练脚本
- 测试方式：运行训练脚本，使用内置数据集
- 设备选择：在 `fuxictr/pytorch/torch_utils.py` 中

### 阶段 2: 迁移方案

采用**条件适配**策略：
- 保留原代码
- 添加 NPU 检测逻辑
- 同时支持 GPU 和 NPU

**需要修改的文件：**
1. `fuxictr/metrics.py` - sklearn 兼容性
2. `fuxictr/pytorch/torch_utils.py` - 设备适配

### 阶段 3: 代码修改

#### 修改 1: sklearn 兼容性

**问题：** 某些 sklearn API 在 NPU 环境中不可用

**修复：**
```python
# fuxictr/metrics.py
# 检查 sklearn 版本，必要时使用兼容写法
import sklearn
print(f"sklearn version: {sklearn.__version__}")
```

#### 修改 2: 设备检测逻辑

**问题：** 原代码只检查 CUDA，不检查 NPU

**修复：**
```python
# fuxictr/pytorch/torch_utils.py
def get_device():
    import torch
    import os

    if os.environ.get("MODEL_CPU_ONLY", "").upper() == "TRUE":
        return torch.device("cpu")

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

### 阶段 4: NPU 验证

**运行命令：**
```bash
export ASCEND_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1
python train.py --config config.yaml
```

**验证结果：**
```
Epoch 1/10: 100%|██████████| 100/100 [00:45<00:00, loss=0.523]
AUC: 0.755208
```

✅ 迁移成功

## 关键经验

1. **大部分 PyTorch 代码天生支持 NPU**
   - 核心的 tensor 操作不需要修改
   - 主要工作量在设备检测逻辑

2. **第三方库兼容性是常见问题**
   - sklearn 版本差异
   - 需要根据实际情况调整

3. **NPU 检测逻辑是关键**
   - 添加 `torch_npu.npu.is_available()` 检查
   - 优先检测 NPU，其次 GPU，最后 CPU

## 总结

| 指标 | 值 |
|------|-----|
| 迁移耗时 | 约 2 小时 |
| 修改文件数 | 2 |
| 代码改动量 | 约 30 行 |
| 验证指标 | AUC 0.755208 |
| 最终状态 | ✅ 成功 |

---

*案例完成日期: 2026-04-03*