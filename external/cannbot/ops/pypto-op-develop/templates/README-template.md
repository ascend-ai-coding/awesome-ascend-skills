# {op_name}

模板说明：本文件是 README.md 的固定模板，由 pypto-op-develop 生成。所有 `{op}` / `{op_name}` / `{...}` 占位符需替换为实际内容。内容面向算子调用方，前 8 节说明 PyTorch 侧 wrapper 接口，后 2 节说明本次生成的交付件。

---

## 算子概述

{一句话描述算子功能与典型使用场景}

## 数学公式

$$
TODO: 填写数学公式
$$

## 接口

```python
def {op}_wrapper(x: torch.Tensor, ...) -> torch.Tensor:
    ...
```

## 参数说明

| 参数 | dtype | shape | 说明 |
|------|-------|-------|------|
| `x` | `{TODO}` | `{TODO}` | 输入张量 |
| 返回值 | `{TODO}` | `{TODO}` | 输出张量 |

## 约束条件

- {TODO: 输入 shape 约束（如支持的维度数、各维度取值限制）}
- {TODO: 输入 dtype 约束}
- {TODO: 硬件/环境约束（如最低 CANN 版本要求）}

## 支持规格

| 项目 | 支持范围 |
|------|---------|
| dtype | `{TODO}` |
| 输入维度 | `{TODO}` |
| 硬件平台 | `{TODO}` |

## 使用示例

```python
import torch
from {op}_impl import {op}_wrapper

# {TODO: 构造输入数据}
x = torch.randn({...}, dtype=torch.float32)

# 调用算子
y = {op}_wrapper(x)

print(y.shape)  # {TODO}
```

## 目录结构

本次生成的文件：

```
.
├── {op}_impl.py      # 算子实现（含 {op}_wrapper 接口）
├── {op}_golden.py    # Golden 参考实现（精度基准）
├── test_{op}.py      # 精度测试入口
├── test_cases.json   # 测试用例配置
└── README.md         # 本文件
```

## 运行方式

```bash
# 设置环境
export TILE_FWK_DEVICE_ID=0

# 运行精度测试（默认遍历所有用例）
python3 test_{op}.py

# 运行单个用例
python3 test_{op}.py case_001

# 列出所有用例
python3 test_{op}.py --list
```
