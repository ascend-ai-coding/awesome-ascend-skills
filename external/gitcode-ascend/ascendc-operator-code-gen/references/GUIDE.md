# AscendC 算子代码生成参考文档加载指南

本文档指导 agent 在不同的算子开发场景下，按需加载对应的 reference 文档，避免一次性加载全部文档造成上下文浪费。

## 参考文档清单

| 文档 | 路径 | 核心内容 |
|------|------|----------|
| 基础数据结构 | `references/basic-data-structures-api.md` | LocalTensor、GlobalTensor、Layout、TPosition 等基础类型 |
| 资源管理 | `references/resource-management-api.md` | TPipe、TQue、TBuf、Double Buffer、Workspace、UB 容量计算 |
| 数据搬运 | `references/data-copy-api.md` | DataCopyPad 用法、Stride 单位、rLength/rLengthAlign、对齐计算 |
| 矢量计算 | `references/vector-compute-api.md` | 标量优化、广播、归约（Level2/Pattern）、Cast 混合精度、Compare |
| 同步控制 | `references/sync-control-api.md` | DMA 异步原理、EnQue/DeQue 同步、PipeBarrier、SyncAll |
| 限制与避坑 | `references/kernel-constraints.md` | 禁止 std::、repeatTime≤255、Compare 256B 对齐、API 黑名单、诊断清单 |

## 场景化加载策略

### 场景 1: Elementwise 算子（ReLU、GELU、Add、Mul 等）

**特征**: 逐元素操作，输入输出 shape 相同，一维切分

**必须加载**:
- `basic-data-structures-api.md` — GlobalTensor/LocalTensor 用法
- `resource-management-api.md` — TPipe/TQue/TBuf 初始化、Double Buffer
- `data-copy-api.md` — DataCopyPad 连续搬运
- `vector-compute-api.md` — 算术/一元/标量运算、Cast 升精度模式
- `kernel-constraints.md` — 禁止 std::、repeatTime 限制

**不需要加载**:
- `sync-control-api.md` — 元素级算子无核间依赖，EnQue/DeQue 已在资源管理中说明

### 场景 2: 归约/归一化类算子（LayerNorm、Softmax、BatchNorm 等）

**特征**: 包含 ReduceSum/ReduceMax、按行/维度切分、可能需要 FP32 中间精度

**必须加载**:
- `basic-data-structures-api.md` — GlobalTensor/LocalTensor
- `resource-management-api.md` — TPipe/TQue/TBuf、UB 容量计算、blockCount 限制
- `data-copy-api.md` — DataCopyPad 多行搬运、rLength/rLengthAlign 用法、Stride 单位
- `vector-compute-api.md` — **重点**: 归约 API（Level2/Pattern）、tmpBuffer 计算、标量优化（Adds/Muls）、Cast 混合精度、多行广播
- `kernel-constraints.md` — repeatTime≤255 分批、Compare 对齐

**不需要加载**:
- `sync-control-api.md` — 行级独立归约无核间依赖

### 场景 3: 池化类算子（AvgPool、MaxPool 等）

**特征**: 滑动窗口操作，多维度遍历，核内有复杂循环结构

**必须加载**:
- `basic-data-structures-api.md` — GlobalTensor/LocalTensor
- `resource-management-api.md` — TPipe/TQue/TBuf、accumBuf 等多临时缓冲区
- `data-copy-api.md` — 多次 DataCopyPad 搬入不同行/位置
- `vector-compute-api.md` — 累加、类型转换、Duplicate 初始化
- `kernel-constraints.md` — 通用 Kernel 限制

**不需要加载**:
- `sync-control-api.md` — 各 slice 独立处理

### 场景 4: 需要核间同步的算子（AllReduce、全局归约等）

**特征**: 多核之间存在数据依赖，需要先局部计算再全局合并

**全部加载**:
- `basic-data-structures-api.md`
- `resource-management-api.md` — 需要 Workspace 管理（GM + UB workspace）
- `data-copy-api.md`
- `vector-compute-api.md`
- `sync-control-api.md` — **重点**: SyncAll/IBSet/IBWait 核间同步、workspace 空间要求
- `kernel-constraints.md`

### 场景 5: 仅修改已有算子（bug 修复、小范围改动）

**按需加载**: 只加载与修改相关的文档。例如：
- 修改计算逻辑 → `vector-compute-api.md` + `kernel-constraints.md`
- 修改数据搬运 → `data-copy-api.md`
- 修改资源分配 → `resource-management-api.md`
- 运行时 crash / 数据错误 → 优先 `kernel-constraints.md` 诊断清单

### 场景 6: 性能优化

**必须加载**:
- `resource-management-api.md` — Double Buffer 原理、UB 利用率
- `vector-compute-api.md` — 标量优化（Adds/Muls 代替 Duplicate+运算）、多行广播、Pattern 归约
- `kernel-constraints.md` — repeatTime 分批优化

## 通用规则

1. **最小加载原则**: 优先加载与当前算子类型直接相关的文档，避免无关文档消耗上下文
2. **限制文档常加载**: `kernel-constraints.md` 包含高频踩坑点，新算子开发时建议始终加载
3. **按编码阶段加载**: 编写 Init 时侧重 `resource-management-api.md`，编写 Compute 时侧重 `vector-compute-api.md`
4. **遇到编译/运行错误时**: 优先查看 `kernel-constraints.md` 诊断清单
5. **数据搬运问题**: 优先检查 `data-copy-api.md` 中的 Stride 单位和 rLength/rLengthAlign 区分
