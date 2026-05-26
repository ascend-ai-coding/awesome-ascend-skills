# Matmul Full-Load 模式深度（A_FULL_LOAD / B_FULL_LOAD）

> **适用架构**：DAV_3510（CANN 9.0.0-beta.2）。L1 容量假设按 L1=512KB 推导。
>
> **实现状态**：A_FULL_LOAD 已在 `references/matmul_custom/` 落地（`matmul_block_mmad_a_full_load.h` + `MatmulTilingAFullLoad`）；B_FULL_LOAD 为对称设计草案，工程模板尚未交付（无 `matmul_block_mmad_b_full_load.h` / `MatmulTilingBFullLoad`），按下文 §3.2 / §4.2 / §4.3 自行实现。
>
> 前置阅读：[`matmul_pattern.md`](matmul_pattern.md) §0 模式总览、§0.5 模板复制、§0.5.3 切换 diff。
>
> 通用全载设计（含性能分析、L1 预算公式、多核排布约束）见 skill `ascendc-performance-best-practices` 的 `reference/matmul/fullload_design.md`。

## 1. 适用判据

Full-Load 核心红利：**小侧矩阵一次载入 L1，跨对侧多 tile 复用**，消除对侧循环中的重复 GM→L1 搬运。

| 模式 | 驻留 L1 的矩阵 | 对侧循环 | 核心条件 |
|------|--------------|---------|---------|
| **A_FULL_LOAD** | A（小 M） | N 向多 tile（`N ≫ M`） | `Align(m,16)·Align(k,16)·sizeof(A) ≤ ~256KB`（L1 留半给 B 双缓冲） |
| **B_FULL_LOAD** | B（小 N） | M 向多 tile（`M ≫ N`） | `Align(n,16)·Align(k,16)·sizeof(B) ≤ ~256KB`（L1 留半给 A 双缓冲） |

**共同条件**：
- 对侧循环次数 `T ≥ 2`（T=1 时无收益）
- 非 StreamK（StreamK 已切 K 给多核，"驻留"语义失效）
- 多核排布满足 `blockCnt ≤ WINDOW_LEN` 且整除 `aicNum`

**反向不适用**：两侧都 > L1/2；对侧 T=1；已启用 StreamK；需要 transA 或 transB。

## 2. L1 布局对比

```
A_FULL_LOAD_MODE:                     B_FULL_LOAD_MODE:
┌──────────────────────┐              ┌──────────────────────┐
│ A 持久区 (offset=0)    │              │ B 持久区 (offset=0)    │
│  = Align(m,16)*       │              │  = Align(n,16)*       │
│    Align(k,16)*       │              │    Align(k,16)*       │
│    sizeof(A)          │              │    sizeof(B)          │
├──────────────────────┤              ├──────────────────────┤
│ B0 (offset=aL1)       │              │ A0 (offset=bL1)       │
├──────────────────────┤              ├──────────────────────┤
│ B1 (offset=aL1+bBuf)  │              │ A1 (offset=bL1+aBuf)  │
└──────────────────────┘              └──────────────────────┘

A 驻留 + B 双缓冲                     B 驻留 + A 双缓冲
```

**L1 预算约束**：

| 模式 | 不等式 |
|------|--------|
| A_FULL_LOAD | `Align(m,16)·Align(k,16)·sizeof(A) + 2·kL1·baseN·sizeof(B) ≤ L1_SIZE` |
| B_FULL_LOAD | `Align(n,16)·Align(k,16)·sizeof(B) + 2·kL1·baseM·sizeof(A) ≤ L1_SIZE` |

Tiling 必须 enforce，违反直接 throw。

## 3. Tiling 算法

### 3.1 A_FULL_LOAD：`MatmulTilingAFullLoad`

```
1. baseM = Align(m, 16)
2. usedCoreNum = min(aicNum, ceil(n/16))
   nPerCore = ceil(n / usedCoreNum)
   baseN = Align(min(nPerCore, 256), 16)
3. baseK = min(Align(k,16),
               FloorAlign(L0A_SIZE/2/sizeof(A)/baseM, 16),
               FloorAlign(L0B_SIZE/2/sizeof(B)/baseN, 16))
4. aL1Reserve = Align(m,16)·Align(k,16)·sizeof(A)
   bL1Avail = L1_SIZE - aL1Reserve
   kL1 = min(baseK * max(1, FloorAlign(bL1Avail/2/(baseN*sizeof(B))/baseK, 1)), Align(k,16))
5. dbL0c = (baseM·baseN·4·2 ≤ L0C_SIZE) ? 2 : 1
```

### 3.2 B_FULL_LOAD：`MatmulTilingBFullLoad`（对称）

```
1. baseN = Align(n, 16)
2. usedCoreNum = min(aicNum, ceil(m/16))
   mPerCore = ceil(m / usedCoreNum)
   baseM = Align(min(mPerCore, 256), 16)
3. baseK = min(Align(k,16),
               FloorAlign(L0A_SIZE/2/sizeof(A)/baseM, 16),
               FloorAlign(L0B_SIZE/2/sizeof(B)/baseN, 16))
4. bL1Reserve = Align(n,16)·Align(k,16)·sizeof(B)
   aL1Avail = L1_SIZE - bL1Reserve
   kL1 = min(baseK * max(1, FloorAlign(aL1Avail/2/(baseM*sizeof(A))/baseK, 1)), Align(k,16))
5. dbL0c = 同 A_FULL_LOAD
```

Init 断言：L1 总量 ≤ L1_SIZE；L0 容量约束。

## 4. BlockMmad 特化关键差异

### 4.1 A_FULL_LOAD vs NO_FULL_LOAD_MODE

| 改动点 | 通用模板 | A 全载 |
|--------|---------|--------|
| **Dispatcher tag** | `MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>` | `MatmulMultiBlockPolicy<A_FULL_LOAD_MODE>` |
| **L1 偏移** | `l1BufferAOffset_[L1_BUFFER_NUM]` 数组 | `l1BufferAOffset_` 单 scalar=0；B pingpong 紧跟 A 之后 |
| **Init 中 A 区** | `aL1OneBuffer_ = baseM*kL1*sizeof(A)` | `aL1Total_ = Align(m,16)*Align(k,16)*sizeof(A)` |
| **状态成员** | 无 | `abL1LoopCnt_` 跨 `operator()` 累计 K-iter |
| **K 循环内 A GM→L1** | 每次都 Copy | `if (abL1LoopCnt_ < kL1Iter_) { Wait; Copy A; Set; }` |
| **内层 L1→L0A** | 从当轮 ping/pong 半区读 | 始终从持久 A 区按 `kL1Offset` 切片 |
| **A 侧 event slot** | 与 B 共用 slot 0/1 | 独立 `A_FLAG=2`（构造 Set 1 次 + 析构 Wait 1 次 = 平衡） |
| **trans 约束** | 支持 transA/transB | 锁 `transA=transB=false` |

### 4.2 B_FULL_LOAD vs A_FULL_LOAD（对称差异）

| 维度 | A_FULL_LOAD | B_FULL_LOAD |
|------|-------------|-------------|
| **驻留矩阵** | A（offset=0） | B（offset=0） |
| **流式矩阵** | B（双缓冲紧跟 A） | A（双缓冲紧跟 B） |
| **独立 event slot** | `A_FLAG=2`（A 侧） | `B_FLAG=2`（B 侧） |
| **跨核切分** | 仅 N 切（mCnt=1） | 仅 M 切（nCnt=1） |
| **尾块禁止** | `mTailTile=1` | `nTailTile=1` |
| **B-NZ 输入** | 不支持（transB=false） | 不支持（transA=false） |

### 4.3 B_FULL_LOAD 的 `operator()` 关键代码

```cpp
// B_FULL_LOAD：B 驻留 L1（对称于 A_FULL_LOAD）
if (abL1LoopCnt_ < kL1Iter_) {
    // B 端：只在首次 K-iter 搬 GM→L1，后续复用
    AscendC::WaitFlag<MTE1_MTE2>(B_FULL_LOAD_B_FLAG);
    auto gmBlockB = gmB.Slice(MakeCoord(kL1Offset, 0), MakeShape(curKL1, curN));
    AscendC::Te::Copy(copyGM2L1, tensorBlockBL1, gmBlockB);
    AscendC::SetFlag<MTE1_MTE2>(B_FULL_LOAD_B_FLAG);
}
// A 端：每次 K-iter 都搬（双缓冲，与通用模板一致）
```

## 5. 实战陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| P1 | runtime hang | baseK 照搬通用模板，被反向约束后超 L0 上限 | 重算 `baseK ≤ L0A_SIZE/2/sizeof(A)/max(baseM,baseN)` |
| P2a | 第二轮 K 卡死 | 守卫内只 Wait 不 Set | 严格成对 `Wait<MTE1_MTE2>(FLAG)` + `Set<MTE1_MTE2>(FLAG)` |
| P2b | 第二次 kernel 调用 hang | 析构漏 WaitFlag → event 计数器泄漏 | 析构必须 `WaitFlag<MTE1_MTE2>(FLAG)`（净平衡） |
| P3 | 多 tile 错误（大面积 mismatch） | 误以为 cmatrixInitVal 在第二次对侧 tile 应为 false | 保持 `(iter0==0 && iter1==0)` —— 每个 tile 独立累加 |
| P4 | scheduler 下溢 | `mCnt=0`（A 全载）或 `nCnt=0`（B 全载） | tiling 保证 count ≥ 1 |
| P5 | bf16 allclose 不过 | A/B 全载与通用模板一样有 ULP 噪声 | MERE+MARE 双门作 PASS/FAIL，allclose 仅参考 |
| P6 | A/B 全载后性能反而劣化 | `T=1` 或 MTE2 假 bound | 严格校验判据，确认真 MTE2 bound |
| P7 | 驻留数据被覆写 | 漏删驻留侧的 `SetFlag<MTE1_MTE2>` | A 全载去掉 A 的释放事件；B 全载去掉 B 的释放事件 |
| P8 | B 全载：A 端 pingpong 用错 event | 未给 A 端独立的 pingpong slot | B 全载下 B 占用 slot 0（独立 B_FLAG），A 端用 slot 0/1 正常 pingpong |

## 6. 模式切换检查清单

### A_FULL_LOAD_MODE 切换

- [ ] launcher 3 行替换（DispatchPolicy / Scheduler / TilingEngine → A_FULL_LOAD_MODE）
- [ ] 锁 `transA=transB=false`
- [ ] 锁 `mTailTile=1`
- [ ] `run.sh` 默认 shape 改为 `M=128 K=256 N=4096 TRANS_B=false`
- [ ] `gen_data.py` 改 `trans_b=false`
- [ ] `verify_result.py` 改用 MERE+MARE 双门
- [ ] tiling 打印 `usedCoreNum/baseM/baseN/baseK/kL1` sanity check

### B_FULL_LOAD_MODE 切换

- [ ] launcher 3 行替换（DispatchPolicy / Scheduler / TilingEngine → B_FULL_LOAD_MODE）
- [ ] 锁 `transA=transB=false`
- [ ] 锁 `nTailTile=1`（**不是** mTailTile——与 A 全载对称相反）
- [ ] `run.sh` 默认 shape 改为 `M=4096 K=256 N=128 TRANS_B=false`
- [ ] `gen_data.py` 改 `trans_b=false`
- [ ] `verify_result.py` 改用 MERE+MARE 双门
- [ ] tiling 打印 sanity check

不满足 L1 容量时 Tiling 直接 throw，提示退回 NO_FULL_LOAD_MODE。

## 参考

- 入口文档：[`matmul_pattern.md`](matmul_pattern.md) §0（模式总览）、§0.5.3（切换 diff）
- 详细设计：skill `ascendc-performance-best-practices` 的 `reference/matmul/fullload_design.md`
