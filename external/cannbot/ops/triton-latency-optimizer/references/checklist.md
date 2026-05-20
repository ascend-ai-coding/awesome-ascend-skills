# 代码规范检查清单

在Ascend NPU上性能高效的triton算子，必须满足以下规范：

## 必须遵循的规范

### 1. 数据类型规范
- [ ] 除非数值精度需要 int64 类型，否则禁止使用 int64 数据类型，必须使用 32 位数值类型进行计算

### 2. 数值比较规范
- [ ] 对于大于等于（>=）、大于（>）、小于等于（<=）、小于（<）四种数值比较操作，在不影响精度的情况下，必须转换成 fp32 数据类型，以启用向量化加速
- [ ] 对于等于（==）、不等于（!=）两种数值比较操作，在不影响精度的情况下，必须转换成 fp32 或 int32 数据类型，以启用向量化加速

### 3. 除法操作规范
- [ ] 对于除法操作，在不影响精度的情况下，必须使用 fp32 或 int32 数据类型进行计算

### 4. 模运算规范
- [ ] 禁止直接使用 `a % b` 操作，必须使用 `a - (a // b) * b` 操作替代

### 5. Grid 并行度规范
- [ ] grid 并行数量禁止超过物理核数：
  - 纯 vector 算子不可以超过 vector 单元数量
  - 既包含矩阵计算及 vector 计算的 mix 算子禁止超过 cube 核数
- [ ] 在任务数量超过核数时，确保获取了正确的核数，且所有核都被用上了
- [ ] 禁止使用多维 grid，仅允许使用一维 grid

获取核心数量的方法：
```python
from typing import Any, Dict, Tuple
import torch
import triton

device = torch.npu.current_device()
device_properties: Dict[str, Any] = (
    triton.runtime.driver.active.utils.get_device_properties(device)
)

num_aicore = device_properties.get("num_aicore", -1)
num_vectorcore = device_properties.get("num_vectorcore", -1)
```

### 6. Task 任务划分规范
- [ ] task 任务划分禁止使用交织划分，每个 grid 任务处理的数据尽可能连续

### 7. 控制流规范
- [ ] 禁止在 triton 代码中使用 `continue` 和 `break` 语句

## 检查流程

1. 加载本文件（checklist.md）
2. 逐一检查上述规范项
3. 如有不满足项 → 修改代码直到满足所有规范
4. 所有规范满足后 → 进行代码验证
