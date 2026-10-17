# GatherMask 使用指南

> **适用场景**：从 interleaved 数据中按固定间隔提取元素时，替代逐元素 SetValue/GetValue 或 Host 侧列拆分。

---

## 目录

- [概述](#概述)
- [模式一：内置固定模式（built-in patterns）](#模式一内置固定模式built-in-patterns)
  - [pattern 1-2：奇偶拆分](#pattern-1-2奇偶拆分)
  - [pattern 3-6：列分量分离](#pattern-3-6列分量分离)
- [模式二：用户自定义模式（custom Tensor mask）](#模式二用户自定义模式custom-tensor-mask)
- [GatherMaskParams 结构](#gathermaskparams-结构)
- [Normal vs Counter 选择](#normal-vs-counter-选择)
- [tile 与循环结构约束](#tile-与循环结构约束-实测经验非官方约束)
- [常见错误](#常见错误)
- [检查清单](#检查清单)
- [适用性检测](#适用性检测)
- [参考资料](#参考资料)

---

## 概述

GatherMask 提供三种用法模式：

| 模式 | `src1Pattern` 类型 | 适用场景 |
|------|-------------------|---------|
| **内置固定模式 1-2** | `uint8_t` = 1 / 2 | 奇偶索引拆分（RoPE 等） |
| **内置固定模式 3-6** | `uint8_t` = 3 / 4 / 5 / 6 | 每 4 元素取 1 个（列分量分离等） |
| **用户自定义模式** | `LocalTensor<uint16_t/uint32_t>` | 非均匀间隔、复杂 mask 逻辑 |

内置固定模式值（910B2C 全部支持）：

| 值 | 二进制 mask | 语义 |
|----|-----------|------|
| 1 | 01010101… | 偶数索引 |
| 2 | 10101010… | 奇数索引 |
| 3 | 00010001… | 每 4 取第 1 |
| 4 | 00100010… | 每 4 取第 2 |
| 5 | 01000100… | 每 4 取第 3 |
| 6 | 10001000… | 每 4 取第 4 |
| 7 | 11111111… | 全取 |

---

## 平台适用性

| 平台 | 适用性 | 说明 |
|------|--------|------|
| **A2 / A3**（DAV_2201） | ✅ 推荐 | 内置固定模式全部可用，是 RoPE 奇偶拆分、列分量分离的推荐方案 |
| **A5**（DAV_3510） | ⚠️ 非首选 | A5 上有更高效的专用指令处理 RoPE 奇偶拆分和 interleaved 列分离，应优先使用专用指令而非 GatherMask |

> 本指南中的模式、参数和约束均基于 A2/A3（DAV_2201）平台验证。A5（DAV_3510）平台开发者应优先查阅对应平台的专用指令文档。

---

## 模式一：内置固定模式（built-in patterns）

### pattern 1-2：奇偶拆分

```cpp
// pattern 1: 偶数索引    pattern 2: 奇数索引
uint64_t rsvdCnt = 0;
LocalTensor<float> src = kvF32[offset];  // 零拷贝偏移

GatherMask(dst, src,
    static_cast<uint8_t>(1), true,               // pattern=1, Counter 模式
    static_cast<uint32_t>(total_elements),       // 元素总数（非 half）
    GatherMaskParams(1, 1, 8, 0), rsvdCnt);
PipeBarrier<PIPE_V>();
```

**约束**：Normal/Counter 均可（官方示例用 Normal pattern=2）。`rsvdCnt` 声明为 `uint64_t` 初始化为 0。`count` 传元素总数。

### pattern 3-6：列分量分离

```cpp
// pattern 3 (00010001): 每 4 取第 1    pattern 4 (00100010): 每 4 取第 2
// pattern 5 (01000100): 每 4 取第 3    pattern 6 (10001000): 每 4 取第 4

// Normal 模式: 每次 repeat 处理 256B = 64 float32 → 16 dst
int32_t alignedLen = (tileLen / 16) * 16;
uint16_t repeatTimes = alignedLen / 16;
GatherMaskParams gmParams(1, repeatTimes, 8, 0);   // stride=8 (256B)，不可写成 1

uint64_t rsvdCnt = 0;
GatherMask(x1, src, static_cast<uint8_t>(3), false, 0, gmParams, rsvdCnt);
GatherMask(y1, src, static_cast<uint8_t>(4), false, 0, gmParams, rsvdCnt);
GatherMask(x2, src, static_cast<uint8_t>(5), false, 0, gmParams, rsvdCnt);
GatherMask(y2, src, static_cast<uint8_t>(6), false, 0, gmParams, rsvdCnt);
PipeBarrier<PIPE_V>();  // GatherMask 是 V 操作，等 V 完成

// 尾块: tileLen 非 16 倍数时标量补齐（≤15 元素，开销可忽略）
for (int32_t j = alignedLen; j < tileLen; ++j) {
    int32_t off = j * 4;
    x1.SetValue(j, src.GetValue(off));
    ...
}
```

**关键约束表**：

| 参数 | 正确值 | 错误值 | 后果 |
|------|--------|--------|------|
| `src0RepeatStride` | `8` (8 DataBlocks = 256B) | `1` | 连续 repeat 重叠 224B → 精度全 fail |
| 对齐粒度 | 16（Normal 每 repeat 输出 16 dst） | 8 | 不匹配 Normal 语义 |
| `repeatTimes` | `alignedLen / 16` | `tileLen / 16`（不向下对齐） | 尾 repeat 源数据不足 256B |
| GatherMask 后 | `PipeBarrier<PIPE_V>()` | 无 barrier | dst 被后续指令读为旧值 |

---

## 模式二：用户自定义模式（custom Tensor mask）

当内置固定模式无法覆盖 mask 逻辑时（如非均匀间隔、不是按固定数量取元素），用 LocalTensor 传入自定义 mask：

```cpp
// src1Pattern: LocalTensor<uint32_t>，每个元素的二进制位 = 1 则选取、0 则跳过
// 例如 0x99999999 = 1001_1001_... → 每 4 元素取第 1 和第 4
LocalTensor<uint32_t> src1Local = inQueueSrc1.DeQue<uint32_t>();

uint32_t mask = 70;          // Counter 模式: 每 repeat 处理 70 个源元素
uint64_t rsvdCnt = 0;

// reduceMode=true (Counter), repeatTimes=2, src0RepeatStride=4 (间隔 4 DataBlocks)
// src1RepeatStride=0: 每次 repeat 复用同一个 src1Pattern
AscendC::GatherMask(dstLocal, src0Local, src1Local,
    true, mask,
    GatherMaskParams(1, 2, 4, 0), rsvdCnt);
```

**关键差异**：

| 特性 | 内置固定模式 | 用户自定义模式 |
|------|------------|--------------|
| `src1Pattern` 类型 | `uint8_t` 常量 | `LocalTensor<uint16_t/uint32_t>` |
| `src1RepeatStride` | 无效（填 0） | 有效，控制 src1 迭代间步长 |
| 数据类型匹配 | — | `half/uint16_t/int16_t` dst → `uint16_t` mask；`float/uint32_t/int32_t` dst → `uint32_t` mask |
| Normal 模式 | ✓ | ✓ |
| Counter 模式 | ✓ | ✓ |

**典型场景**：需要自定义间隔（如每 5 取 2、非均匀采样）或 mask 值运行时决定时使用。

---

## `GatherMaskParams` 结构

```cpp
struct GatherMaskParams {
    uint8_t  src0BlockStride;   // 单 repeat 内 DataBlock 步长，连续=1
    uint16_t repeatTimes;       // Normal: 256B 块数；Counter: 与 mask 配合
    uint16_t src0RepeatStride;  // 相邻 repeat 起始地址间隔（DataBlock 单位）
    uint8_t  src1RepeatStride;  // 内置固定模式无效(填0)；用户自定义模式有效
};
// GatherMaskParams(src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride)
```

Normal 模式：`repeatTimes = 总源元素数 / 64`，`src0RepeatStride = 8`（64 float × 4B / 32B = 8 DataBlocks）。repeatTimes=1 时 `src0RepeatStride` 可填 0。

---

## Normal vs Counter 选择

| 模式 | `reduceMode` | 数据量控制 | 推荐场景 |
|------|-------------|-----------|---------|
| **Normal** | `false` | 每次 repeat 固定 256B，`mask` 无效 | built-in patterns 3-6（推荐，语义简洁） |
| **Counter** | `true` | 每次 repeat 处理 `mask` 个元素 | pattern 1-2（RoPE 奇偶拆分）、用户自定义模式、built-in patterns 3-6（也可用） |

两种模式均可搭配 built-in patterns 3-6。Normal 模式语义更清晰（固定 256B/repeat），优先推荐。

> **官方约束**（CANN 8.5.0）：GatherMask 调用后接口内部自动重置为 Normal 模式。若混用 Counter 模式，调用后需显式重新设置。

---

## tile 与循环结构约束（※ 实测经验，非官方约束）

**tile 大小**：tile 过小会导致 DMA 次数随 N 线性增长，压过 GatherMask 收益。**规则**：tile 填到 UB 上限。910B2C UB=248KB，GatherMask + compute buffers ≈112KB，tile ≥ 2048 安全。

**循环结构**：解交织维度必须放外层循环。

```
✓ 解交织维度 OUTER — 解交织一次，内层全量复用  →  DMA O(N/tile)
✗ 解交织维度 INNER — 每外层迭代重解交织        →  DMA O(M×N/tile²)
```

外层循环的数据 UB 常驻。小维度走标量 GM 读取即可，不要两侧都 GatherMask——增加 UB 压力无额外收益。

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|-----|------|---------|
| 精度全 fail，误差 > 0.3 | `src0RepeatStride=1`，连续 repeat 重叠 224B | 设为 `8`（8 DataBlocks = 256B） |
| 混用模式后计算结果异常 | GatherMask 调用后自动切回 Normal | Counter 模式下调用后显式重新设置 |
| 大 N 加速比退化到 <1x | tile < 512 且解交织维度放内层循环 | tile ≥ 2048 + 解交织维度放外层 |
| 尾块数据错乱 | `repeatTimes = tileLen/16` 未向下对齐 | `alignedLen = (tileLen/16)*16`，尾块标量 |
| dst 数据被读为旧值 | GatherMask 后未加 barrier | 添加 `PipeBarrier<PIPE_V>()` |
| 用户自定义模式 dst 类型与 mask 类型不匹配 | half dst 配 uint32_t mask | half/uint16_t → uint16_t mask；float/uint32_t → uint32_t mask |

---

## 检查清单

**参数配置**：
- [ ] `reduceMode` 与模式匹配（patterns 1-2 Normal/Counter 均可，patterns 3-6 推荐 Normal）
- [ ] `src0RepeatStride = 8`（Normal 模式 repeatTimes > 1 时）
- [ ] `repeatTimes = alignedLen / 16`（内置固定模式，向下对齐到 16）
- [ ] `rsvdCnt` 声明为 `uint64_t` 并初始化为 0
- [ ] 用户自定义模式：mask Tensor 数据类型与 dst 匹配

**tile 与循环**：
- [ ] tile ≥ 2048（填到 UB 上限）
- [ ] 解交织维度放外层循环
- [ ] 不同时对两侧做 GatherMask（小维度走标量）

**同步**：
- [ ] GatherMask 后 `PipeBarrier<PIPE_V>()`
- [ ] Counter 模式混用时注意自动切回 Normal 行为

---

## 适用性检测

```bash
# Host 侧列拆分 → 可用 built-in patterns 替代
grep -nE "\.select\(.*\)\.contiguous\(\)" op_host/*.cpp

# 逐元素 SetValue/GetValue 解交织 → 可用内置固定模式或用户自定义模式替代
grep -nE "for.*SetValue.*GetValue|for.*GetValue.*SetValue" op_kernel/*.cpp

# 检查现有 GatherMask 参数是否正确
grep -nE "GatherMaskParams.*\{1.*repeatTimes.*1.*0\}" op_kernel/*.cpp      # stride=1，错误
```

---

## 参考资料

- CANN 8.5.0 API 文档 — GatherMask：`https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0071.html`
- P100 卡片：GatherMask 替代逐元素 SetValue/GetValue（RoPE 奇偶拆分场景）
- P101 卡片：GatherMask 替代 Host 侧列拆分
