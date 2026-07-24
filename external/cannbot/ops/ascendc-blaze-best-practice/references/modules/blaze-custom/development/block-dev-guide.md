# Block 层扩展开发指南

> **适用路径**：blaze_custom（路径 A）

---

## §1 BlockMmad 接口规范

### 模板签名（8 参数 + SFINAE，`block_mmad.h`）

```cpp
template <class DispatchPolicy_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class Enable = void>
class BlockMmad;
```

blaze 库为 10 参数（多 `BiasType_` + `LayoutBias_`），见 `references/fundamentals/blaze-framework-overview.md` §4.4。

### 必需内部类型

```cpp
using L0CType = conditional_t<is_same_v<AType, int8_t>, int32_t, float>;
static constexpr bool transA = TagToTrans<LayoutA>::value;
struct Params   { GM_ADDR aGmAddr; GM_ADDR bGmAddr; GM_ADDR cGmAddr; };
struct L1Params { uint64_t kL1; };
```

### 接口合约

| 接口 | 职责 |
|------|------|
| `BlockMmad()` | 预发所有 event slot |
| `~BlockMmad()` | 排空所有 event slot |
| `Init(TupleShape, BlockShape, L1Params, bool)` | 计算 L1 偏移、K 迭代次数 |
| `operator()(TensorA, TensorB, TensorC, BlockShape)` | 执行完整 K 迭代流水 |

---

## §2 K-L1/K-L0 双层循环骨架

```cpp
for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
    uint64_t l1BufId = abL1LoopCnt_ & L1_BUFFER_MASK;
    auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;

    WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);
    Copy(copyGM2L1, tensorAL1, gmA.Slice(...));
    Copy(copyGM2L1, tensorBL1, gmB.Slice(...));
    SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
    WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);

    for (uint16_t iter1 = 0; iter1 < CeilDiv(curKL1, baseK_); ++iter1) {
        uint64_t l0BufId = l0PingPong_ & 0x1;
        WaitFlag<HardEvent::M_MTE1>(l0BufId);
        Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        Copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        SetFlag<HardEvent::MTE1_M>(l0BufId);
        WaitFlag<HardEvent::MTE1_M>(l0BufId);

        bool isLast = (iter0 + 1 == kL1Iter_) && (iter1 + 1 == kL0Iter);
        bool isFirst = (iter0 == 0 && iter1 == 0);
        MmadParams params{(uint16_t)curM, (uint16_t)curN, (uint16_t)curKL0,
            isLast ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION, isFirst};
        Mmad(mmadAtom, tensorL0C, tensorAL0, tensorBL0);
        SetFlag<HardEvent::M_MTE1>(l0BufId);
        l0PingPong_++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
    abL1LoopCnt_++;
}
```

---

## §3 事件管理（Init-Drain 模式）

构造预发、析构排空，确保事件平衡：

```cpp
BlockMmad() {
    for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i)
        SetFlag<HardEvent::MTE1_MTE2>(i);
    SetFlag<HardEvent::M_MTE1>(ZERO_FLAG);
    SetFlag<HardEvent::M_MTE1>(FIRST_FLAG);
    SetMMLayoutTransform(true);
}
~BlockMmad() {
    for (uint8_t i = 0; i < L1_BUFFER_NUM; ++i)
        WaitFlag<HardEvent::MTE1_MTE2>(i);
    WaitFlag<HardEvent::M_MTE1>(ZERO_FLAG);
    WaitFlag<HardEvent::M_MTE1>(FIRST_FLAG);
    SetMMLayoutTransform(false);
}
```

构造 `SetFlag` 与析构 `WaitFlag` 必须完全对称。遗漏析构 WaitFlag → 第二次 kernel 调用 hang。

> 完整配对规则 → `references/fundamentals/blaze-sync-patterns.md` §3

---

## §4 tensor_api 必要知识

### MakeCopy / Copy（dst 在前）

```cpp
auto copyGM2L1 = MakeCopy(CopyGM2L1{});
Copy(copyGM2L1, dstL1Tensor, srcGMTensor);
```

13 种 CopyOperation：`CopyGM2L1`、`CopyL12L0A`、`CopyL12L0B`、`CopyL0C2GM`、`CopyL0C2UB` 等。Routing 按 `(dstLoc, srcLoc, dstPattern, srcPattern)` 编译期派发。

### MakeMmad / Mmad

```cpp
MmadParams params{m, n, k, unitFlag, cmatrixInitVal};
Mmad(MmadAtom<MmadTraits<MmadOperation>>{}.with(params), dstL0C, srcL0A, srcL0B);
```

`unitFlag`：`FINAL_ACCUMULATION`（K 末条）/ `NON_FINAL_ACCUMULATION`。`cmatrixInitVal`：`true` = 清零再累加（K 首条）。

> 完整 API → `references/fundamentals/tensor-api-reference.md` §2.4-§2.5

---

## §5 Layout 与数据通路

### L1 双缓冲布局

```
L1: [A0|B0] | [A1|B1]    （half0 / half1 ping-pong）
```

偏移：`l1BufferAOffset_[bufId] = (bufId & 1) * l1HalfSize + (bufId >> 1) * aL1OneBuffer_`

### GM→L1 数据通路

| GM Pattern | L1 Pattern | 说明 |
|------------|------------|------|
| `NZLayoutPtn` / `ZNLayoutPtn` | 同 GM | 块拷贝 |
| `NDExtLayoutPtn`（trans=false） | `NZLayoutPtn` | ND→NZ 转换 |
| `DNExtLayoutPtn`（trans=true） | `ZNLayoutPtn` | DN→ZN 转换 |

L0 固定：L0A = `NZLayoutPtn`，L0B = `ZNLayoutPtn`，C0 = 16。

> Layout 详解 → `references/fundamentals/blaze-matmul-layout.md`
