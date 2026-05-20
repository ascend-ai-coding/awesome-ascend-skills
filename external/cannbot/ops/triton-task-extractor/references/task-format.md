# 算子任务格式规范

算子任务文件支持两种模式：

- **单 case 模式**：`.py` 自包含，`get_inputs()` 返回单组输入
- **多 case 模式**：`.py` + 同名 `.json` 配对，`get_input_groups()` 在运行时读取 `.json` 构造多组输入

下游脚本（`verify.py` / `benchmark.py`）已内建判断：优先调用 `get_input_groups()`，回落 `get_inputs()`。

---

## 模式 A：单 case

### 文件结构

单一自包含 Python 文件，包含以下四个必需部分。

#### 1. Imports 区

```python
import torch
import torch.nn as nn
# 只允许标准库和 PyTorch 相关包
# 禁止 import 项目内的其他文件
```

#### 2. Model 类

```python
class Model(nn.Module):
    def __init__(self, <init_params>):
        super(Model, self).__init__()
        # 保存所有初始化参数

    def forward(self, <forward_inputs>) -> torch.Tensor:
        # 核心计算逻辑
        return output
```

#### 3. `get_inputs()` 函数

```python
def get_inputs():
    """返回 forward() 的输入参数列表"""
    input1 = torch.randn(batch_size, dim)
    input2 = torch.randn(batch_size, dim)
    return [input1, input2]
```

#### 4. `get_init_inputs()` 函数

```python
def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    return [dim_value]
```

---

## 模式 B：多 case

适用于需要在多种 shape / dtype 组合下评测同一算子的场景（典型来源：`benchmarks/level430/`）。

### 文件配对

```
{op_name}.py     # 自包含 Python：Model + get_input_groups + get_init_inputs
{op_name}.json   # JSONL：每行一个 case 的输入规格，与 .py 同目录同名
```

`.py` 内通过 `os.path.dirname(__file__)` 定位同目录 JSON。**两个文件必须同时复制到工作目录**，路径关系不可改。

### `.json` Schema (JSONL，每行一个 case)

```jsonc
{"inputs": [
  {"name": "x", "type": "tensor", "required": true, "dtype": "int8", "shape": [16, 20]},
  {"name": "y", "type": "tensor", "required": true, "dtype": "int8", "shape": [16, 20]}
]}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 参数名，对应 `forward()` 形参 |
| `type` | str | `tensor` / `attr`（标量属性）/ `tensor_list` |
| `dtype` | str | `float32`/`float16`/`bfloat16`/`bool`/`int8`/`int16`/`int32`/`int64`/`uint8`/`float64`/`complex64`/`complex128`/`str` |
| `shape` | list[int] | 张量形状（`type=tensor` 时必填，`[]` 表示 scalar tensor） |
| `value` | any | `type=attr` 时，标量属性值 |
| `required` | bool | 是否必填（一般 true） |

### `.py` 模板

```python
import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(x, y)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "{op_name}.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
        "bool": torch.bool, "int8": torch.int8, "int16": torch.int16,
        "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8,
        "float64": torch.float64, "complex64": torch.complex64, "complex128": torch.complex128,
    }

    def make_tensor(info):
        dtype = dtype_map[info["dtype"]]
        shape = info["shape"]
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch.bool)
        elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return torch.randint(-10, 10, shape, dtype=dtype)
        elif dtype == torch.uint8:
            return torch.randint(0, 10, shape, dtype=dtype)
        elif dtype in (torch.complex64, torch.complex128):
            return torch.complex(torch.randn(shape), torch.randn(shape)).to(dtype)
        else:
            return torch.randn(shape, dtype=dtype)

    input_groups = []
    for case in cases:
        group = []
        for spec in case["inputs"]:
            if spec["type"] == "tensor":
                group.append(make_tensor(spec))
            elif spec["type"] == "attr":
                group.append(spec["value"])
        input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
```

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 所有依赖函数必须内联到 `.py` 中；多 case 模式下，外部依赖**仅允许**同名 `.json` |
| 可执行（单 case） | `Model(*get_init_inputs())(*get_inputs())` 必须直接运行 |
| 可执行（多 case） | `Model(*get_init_inputs())(*get_input_groups()[i])` 对所有 `i` 都必须直接运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 合理输入 | 输入规模不能过小或过大 |
| 一致返回 | 返回类型/形状必须与原始实现一致 |
| 模式选择 | **禁止**将多 case 源（含 `get_input_groups`）降级为单 case 任务文件 |
