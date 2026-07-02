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

### 7. 循环索引计算规范
- [ ] 在 `for i in range(N)` 展开循环内，**禁止**使用可变累加器更新索引偏移量，必须从循环变量 `i` 独立计算每轮索引

**错误写法**（可变累加器，创建跨迭代数据依赖，编译器无法并行化展开体，性能大幅劣化）：
```python
off_block = tl.arange(0, BLOCK)
for i in range(NUM_BLOCK):
    # ... 使用 off_block 访存 ...
    off_block = off_block + BLOCK  # 禁止：迭代间真实数据依赖
```

**正确写法**（每轮从 `i` 独立计算，无迭代间依赖）：
```python
off_block_base = tl.arange(0, BLOCK)
for i in range(NUM_BLOCK):
    row_off = i * BLOCK
    off_block = row_off + off_block_base  # 每轮独立，编译器可并行优化
    # ... 使用 off_block 访存 ...
```

**原理**：Triton Ascend 编译器对 `tl.constexpr` 约束的循环进行展开。可变累加器 `x = x + C` 在展开体中产生跨迭代的真实数据依赖链，导致展开体被迫串行执行。改为 `x = i * C + base` 后每次迭代的索引仅依赖循环计数器 `i`，编译器可消除依赖、并行调度展开体。

### 8. 控制流规范
- [ ] 禁止在 triton 代码中使用 `continue` 和 `break` 语句

## 检查流程

1. 加载本文件（checklist.md）
2. 逐一检查上述规范项
3. 如有不满足项 → 修改代码直到满足所有规范
4. 所有规范满足后 → 进行代码验证
