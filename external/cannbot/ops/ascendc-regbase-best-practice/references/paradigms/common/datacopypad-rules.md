# DataCopyPad 规则

> DataCopyPad、stride 单位、padding 陷阱。

## 1 函数原型与参数结构体

```cpp
// GM → UB（MTE2）：padParams 不可省
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst,
                                                        const GlobalTensor<T>& src,
                                                        const DataCopyExtParams& dataCopyParams,
                                                        const DataCopyPadExtParams<T>& padParams);

// UB → GM（MTE3）：无 padParams
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst,
                                                        const LocalTensor<T>& src,
                                                        const DataCopyExtParams& dataCopyParams);
```

> **stride 两种语义**：`DataCopyExtParams` 的 stride 是 **gap**（块尾→下一块块头）；`LoopModeParams` 的 stride 是 **advance**（块头→下一块块头）。

### DataCopyExtParams

```cpp
struct DataCopyExtParams {
    uint16_t blockCount;     // [1, 4095]
    uint32_t blockLen;       // [1, 2097151] byte
    int64_t  srcStride;      // gap 语义. 单位：GM 侧 byte，GM 支持负值；UB 侧 datablock(32B)，不允许负值
    int64_t  dstStride;      // 同上
    uint32_t rsv;            // 必须显式填 0
};
```

**注意**:
- blockCount 不能取0，有效范围 [1, 4095]
- blockLen 不能取0，有效范围 [1, 2097151]

### DataCopyPadExtParams<T>

```cpp
template <typename T>
struct DataCopyPadExtParams {
    bool    isPad;
    uint8_t leftPadding;     // 元素个数（非 byte），字节数 ≤ 32B
    uint8_t rightPadding;    // 元素个数（非 byte），字节数 ≤ 32B
    T       paddingValue;    // 64-bit T 时只能填 0
};
```

### LoopModeParams

```cpp
struct LoopModeParams {
    uint32_t loop1Size;        // [0, 2^21)
    uint32_t loop2Size;        // [0, 2^21)
    uint64_t loop1SrcStride;   // byte（advance 语义）
    uint64_t loop1DstStride;   // byte（advance 语义），UB 侧须 32B 对齐
    uint64_t loop2SrcStride;   // byte（advance 语义）
    uint64_t loop2DstStride;   // byte（advance 语义），UB 侧须 32B 对齐
};
```

**注意**:
- stride 范围：GM 侧 `[0, 2^40)`; UB 侧 `[0, 2^21)` 且必须 32B 对齐。
- `loop2Size` 不使用时填 **1**（填 0 = 整批搬运 0 次）

**loop mode 本质**：在 DataCopyPad 单次调用的基础上，额外提供两层硬件自动循环（loop1 内层 × loop2 外层），替代软件 for 循环。

**使用流程**（三步）：

```cpp
// 1. 使能 loop mode 并设置参数
SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
// 2. 调 DataCopyPad（硬件自动按 loop 参数迭代）
DataCopyPad<T>(dstLocal, srcGlobal, copyParams, padParams);
// 3. 复位寄存器（不调则残留，影响下次同方向搬运）
ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
```

**DataCopyMVType**：搬运方向枚举。

| 枚举值 | 方向 | src 侧 | dst 侧 |
|--------|------|--------|--------|
| `OUT_TO_UB` | GM → UB（MTE2） | GM | UB |
| `UB_TO_OUT` | UB → GM（MTE3） | UB | GM |

方向决定 loop stride 4 个字段哪一侧是 UB（须 32B 对齐、范围 [0, 2^21)）、哪一侧是 GM（范围 [0, 2^40) 无对齐要求）。

## 2 stride 语义与对齐规则

### stride 单位速查

| 字段 | 单位 |
|------|------|
| `DataCopyExtParams.blockLen` | **永远 byte** |
| `DataCopyExtParams` stride（GM 侧） | byte |
| `DataCopyExtParams` stride（UB 侧） | **datablock = 32B** |
| `LoopModeParams.*Stride`（无论方向） | **永远 byte** |
| `DataCopyPadExtParams.leftPadding/rightPadding` | **元素个数** |

> **方向决定 src/dst 对应哪一侧**：MTE2 (GM→UB) 时 src=GM(byte)、dst=UB(datablock)；MTE3 (UB→GM) 时 src=UB(datablock)、dst=GM(byte)。

### ⚠️ 硬件对齐规则（易错点）

**起始地址对齐**：UB 侧源/目的起始地址必须 32B 对齐，GM 侧不强制。

DataCopyPad 硬件将每个块（burst）在 **UB 侧**占用的空间自动对齐到 blockSize（32B）：

```
UB 块占用空间 = CeilAlign(blockLen + (leftPadding + rightPadding) × sizeof(T), 32B)
```

**`DataCopyExtParams` 的 src/dst stride 都是 gap（块间空隙），不是 stride（块间步距）**（注意：`LoopModeParams` 的 stride 是 advance 语义）。UB 侧和 GM 侧 stride 的区别在于单位：

| 侧 | stride 单位 | stride=0 含义 |
|:---|:-----------|:-------------|
| UB 侧 | 32B (datablock) | 块间无额外空隙，下一块紧跟**上一块 32B 对齐后的位置** |
| GM 侧 | byte | 块间无额外空隙（dense） |

> ⚠️ src/dst 对应的侧取决于方向：
> - **MTE2 (GM→UB)**：srcStride = GM 侧 (byte)，dstStride = UB 侧 (32B gap)
> - **MTE3 (UB→GM)**：srcStride = UB 侧 (32B gap)，dstStride = GM 侧 (byte)
>
> 典型错误：把 UB 侧 stride 理解为步距，认为 stride=0 会导致块重叠。实际上 stride 表示的是"块尾"到"下一块块头"之间的额外空隙。stride=0 是合法且常见的正确值。

### ⚠️ Loop 迭代重叠约束

Loop 模式下各层 stride 必须保证迭代间 UB 写入区域不重叠。注意 `DataCopyExtParams.dstStride` 在 UB 侧单位是 **datablock(32B)**，而 `LoopModeParams.loop*DstStride` 单位是 **byte**，比较前需统一单位：

```
loop1DstStride ≥ blockCount × (CeilAlign(blockLen, 32B) + dstStride × 32)
loop2DstStride ≥ loop1Size × loop1DstStride
```

其中 `CeilAlign(blockLen, 32B)` 是单个 block 在 UB 侧的实际占用，`dstStride × 32` 把 datablock 单位的 gap 转为字节。

### ⚠️ GM 地址范围硬约束

MTE2/MTE3 操作 GM 时，地址必须在 tensor 的有效数据范围内，**超过 1 Bytes 都不行**。

## 3 Padding 行为（关键陷阱）

### Padding 行为分支

`blockLen + (leftPadding + rightPadding) × sizeof(T)` 是否满 32B 决定两条分支：

### 满足 32B 对齐

| `isPad` | 左右填充值 |
|---------|-----------|
| `false` | 随机值 |
| `true`  | `paddingValue` |

### 不满足 32B 对齐

框架自动补 dummy 字节凑齐 32B：

| leftPadding/rightPadding | isPad | 左右填充 + dummy 的值 |
|--------------------------|-------|---------------------|
| 都为 0 | 任意 | **dummy = 搬运数据块的第一个元素值** ← ⚠️ 关键陷阱 |
| 任一非 0 | `false` | 全部随机值 |
| 任一非 0 | `true`  | 全部 `paddingValue` |

**⚠ `isPad=true` 不足以绕过首元素值陷阱**：当 `leftPadding = rightPadding = 0` 时，无论 `isPad` 取值，dummy 都填首元素值。要让 dummy 走 `paddingValue` 路径，**必须 `rightPadding` 非 0**。

**标准做法**：要让 dummy 填 `paddingValue`（isPad=true 时），`rightPadding` 必须非 0，标准取值 `(32 − blockLen % 32) / sizeof(T)`。rightPadding 不参与行宽对齐——UB 侧实际占用恒为 `CeilAlign(blockLen, 32B)`，由 HW 自动补齐。

### UB → GM 方向

不需要 padParams。UB 侧 blockLen 不到 32B 时，框架自动补 dummy 让 UB 读出对齐；搬到 GM 时 dummy 被丢弃。

## 4 常见陷阱

| 陷阱 | 修正 |
|------|------|
| **dummy 默认填首元素值** | `rightPadding=0` 时无论 `isPad` 取值，dummy 都填首元素值。要让 dummy 走 `paddingValue`，必须 `rightPadding` 非 0 且 `isPad=true` |
| **GM 侧 stride 单位混淆** | `DataCopyExtParams` 里 GM 端 stride 永远 byte，UB 端永远 datablock；`LoopModeParams` 里 stride 永远 byte |
| **leftPadding/rightPadding 单位** | 填**元素个数**，不是 byte。字节数 = 元素数 × `sizeof(T)`，且不能超 32B |
| **漏调 ResetLoopModePara** | 寄存器残留，下一次同方向 DataCopyPad 也走 loop 模式 |
| **64-bit dtype paddingValue 非 0** | `int64_t/uint64_t/double` 时 `paddingValue` 只允许 0 |
| **Loop 迭代重叠** | 见 §2「Loop 迭代重叠约束」 |
| **UB 内逻辑轴数超 4** | 超出用 for 循环拆外层 |
| **DataCopyExtParams 漏填 rsv** | **永远显式末尾补 0**：`{count, len, srcStride, dstStride, 0}` |
