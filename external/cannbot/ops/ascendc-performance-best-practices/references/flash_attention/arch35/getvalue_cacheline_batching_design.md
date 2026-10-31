# GetValue DataCache 批量读取优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

## 1. 优化目标

针对 sparse attention / SCFA hot loop 中连续读取 topk index、block id、row id 等 GM metadata 的场景，将每次 1-2 个 `GlobalTensor::GetValue` 改为 cache-line friendly 的小批量读取，再按原来的语义粒度消费，提升 DataCache 局部性并缓解 Scalar / SCALARLDST 访存压力。

## 2. 架构概览

### 优化前

```text
for each sparse chunk:
  GetValue(topk[k])
  GetValue(topk[k + 1])
  GetkeyOffset(token0)
  GetkeyOffset(token1)
  DataCopyPad pair
```

问题：topk index 是连续 GM 数据，但每次只读 2 个，随后穿插 block table 读取、offset 计算和搬运参数构造，容易低效消费同一条 DataCache line。

### 优化后

```text
for each sparse chunk:
  tokenData[0..7] = consecutive GetValue(topk[k..k+7])
  for i in {0, 2, 4, 6}:
    GetkeyOffset(tokenData[i])
    GetkeyOffset(tokenData[i + 1])
    DataCopyPad pair
```

关键点：读取阶段批量化，下游 KV copy 仍保持原来的 2 token 一组逻辑，减少行为变化范围。

## 3. 关键参数 / 修改点

| 参数 / 模块 | 优化前 | 优化后 | 说明 |
|-------------|--------|--------|------|
| sparse-index batch size | 2 | 8 | 8 是参考模式的批量值，不代表当前 SFA 已验证获益；按 dtype / 寄存器压力复测 |
| `GetRealCmpS2Idx` | 两个引用返回值 | `int32_t tokenData[8]` | 连续读取 topk index |
| `ProcessSparseKv` | `s2 += 2` | `s2 += 8` | 单次收集 4 组 pair |
| `CopyInKvSparse` | 单 pair 处理 | 内部按 `i += 2` 处理 4 个 pair | 保持原 pairwise stride / fallback 逻辑 |
| tail 语义 | 逐 pair early-exit | `tokenData` 预填 `-1`，尾部不足保持 early-exit | 保持边界行为 |

## 4. 核心计算循环

### 优化前

```cpp
int64_t token0Idx;
int64_t token1Idx;
GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
s2 += 2;
dealRow += CopyInKvSparse(kvInUb, dealRow,
                          token0Idx, token1Idx,
                          runInfo, constInfo);
```

### 优化后

```cpp
int32_t tokenData[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
GetRealCmpS2Idx(tokenData, s2, runInfo, constInfo);
s2 += 8;
dealRow += CopyInKvSparse(kvInUb, dealRow,
                          tokenData,
                          runInfo, constInfo);
```

`CopyInKvSparse` 内部继续按两个 token 一组执行 offset / stride / fallback copy：

```cpp
for (uint32_t i = 0; i < 8; i += 2) {
    if (tokenData[i] < 0 || tokenData[i + 1] < 0) {
        break;
    }
    ...
}
```

## 5. 效果与适用边界

| 场景 | 建议 |
|------|------|
| 同一个 GM metadata 表存在连续或近连续 `GetValue` | 适用 |
| 每 1-2 个 `GetValue` 后会穿插其他 Scalar GM 访问 | 适用 |
| metadata 自身的读取地址随机分散、空间局部性弱 | 不适用；这里说的是索引表地址，不是表中存储的 KV token 值 |
| 扩大 batch 会读取大量无用数据 | 不适用 |
| 批量数组导致寄存器 spill 或栈访存压力上升 | 回退 batch size 或放弃 |

效果：减少连续 sparse index 读取与其他 Scalar GM 访问的交错，提高 DataCache line 利用率。该案例未提供具体量化数据，因此不编造收益数字；落地时以 main/control shape、精度、Scalar/SCALARLDST counters 作为验收。

连续读取 8 个 int32 只说明覆盖 32 字节，不能据此认定恰好是一条目标硬件的 cache line，也不能认定生成了一条向量 load。先核对实际读取地址、对齐与生成代码。表中 token 值即使随机，metadata 读取仍可能连续；只有进一步合并 KV DMA 时，才需要验证 token 对应的物理地址规律。批读与 DMA 合并是两个独立手段，应分别测量。

## 6. 与当前SFA baseline的关系

arch35 `SparseFlashAttention` 的当前baseline在
`GetRealCmpS2Idx → GetkeyOffset → CopyInKvSparse` 链中每次读取两个连续index，
并尝试用一条 `DataCopyPad blockCount=2` 搬运对应KV。本文的batch=8方案只改变
metadata读取节奏，不应改写下游pairwise搬运及fallback语义。

该baseline的完整流程、内存布局、真机case和必要片段见
[token-wise SFA实例](example-sparse-flash-attention-tokenwise-mla.md)。现有实测验证的是未批量化baseline；
batch=8仍是待实施候选，不能将baseline的精度或性能数字记作本优化收益。
