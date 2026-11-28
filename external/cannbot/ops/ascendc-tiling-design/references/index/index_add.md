# P6: 沿轴索引累加（IndexAdd）

> 状态：已补充（2026-09-22，基于 IndexAdd 算子 910B（dav_c220）实战验证）
> 硬件基线：Ascend 910B3 / dav_c220 / 40 个 AIV core / 单核 UB 192KB / CANN 9.0.0
>
> **性能状态：全部反超原生 torch（1.04x~17.50x，**无短板**）——
> 此前 `mode0 4096×4096 dim0 n4096` 的 0.33x 已由 per-core 命中表消除，见 §5**

## 适用场景

- 沿指定轴 `dim` 把 `updates` 按 1-D index 累加到输入：`out[o, idx[k], i] += alpha * upd[o, k, i]`
  - `outer = prod(shape[:dim])`，`axis = shape[dim]`，`inner = prod(shape[dim+1:])`
  - `index` 长度 `nidx`，`updates` 形状 `(outer, nidx, inner)`（与输入同 rank，仅 dim 维换成 nidx）
- 典型算子：`torch.index_add` / `aclnnIndexAdd`
- 与 P5（IndexSelect）共享 `outer/axis/inner` 分解，但语义上是**读-改-写**：
  多个 k 可能命中同一行 → 同一 slab 行需要**多次累加**

## 核心约束（决定整个方案形态）

**不同核不能并发写同一个输出元素。**

直觉方案是累加时用 `SetAtomicAdd<T>()` 让硬件保证一致性，但**实测在 `dav_c220` 上不可用**：

```cpp
// ⚠️ 看似正确，实测在 dav_c220 上不可用
SetAtomicAdd<float>();
DataCopyPad(outputGm[outOffset], srcLocal, copyParams);  // 期望原子累加到 GM
SetAtomicNone();
```

`dav_c220` 的 GM 原子加要求目标地址 **32B 对齐**，而目标偏移 `(o*axis+t)*inner`
在 `inner` 不是 8 的倍数（fp32 下 32B = 8 元素）时**不保证对齐** → 原子加失败或结果错误。
**同样地，`scatter_add` / `scatter_reduce` 的累加也不要依赖 `SetAtomicAdd`**（见 [scatter.md](scatter.md) §3）。

因此分核必须保证**区间互不重叠**：

> 工作单元 = `(o, tTile, iTile)`，覆盖 `t ∈ [t0, t0+tLen)`、`i ∈ [i0, i0+iLen)`。
> 不同单元的 `(t 区间, i 区间)` 互不重叠 → 任意两核不会写同一输出元素，
> **无需原子、结果确定**（与 P3/P4 scatter 的「输出元素单属主」结论同源）。

**副产品（重要）**：`units = outer × tTiles × iTiles` **恰好覆盖整个输出空间**，
每个输出元素恰好被一个单元写一次。因此 kernel 可以**全量写出 y**，
调用方用 `empty_like` 即可，**不需要先 `x.clone()`**。

### 读 x 写 y，而不是原地改 var（两条收益）

```
Init(x, index, updates, alpha, y, tiling)   # 5 个输入指针
out = at::empty_like(input)                 # torch 侧：无需 clone
```

1. **省 1x 全量带宽**：原地改 var 需要 `var.clone()` 先做一次设备拷贝（读+写全量）
2. **避开 clone 对 kernel 不可见的竞态**：`clone` 是 torch 算子，会 enqueue 进 torch_npu
   延迟任务队列；若取 stream 的时机早于它，kernel 可能读到 clone 之前的旧数据

## 核心思路

### 1. 任意 dim 直接算，**不需要 transpose**（同 P5，与 P2 的关键差异）

index 是 1-D，输出与输入同 rank（仅 `dim` 维换成 `nidx`），内存布局天然连续，
kernel 只用**与 dim 无关**的寻址公式（源 `var[(o*axis+t)*inner+i]`、
`upd[(o*nidx+k)*inner+i]`、目标 `y[(o*axis+t)*inner+i]`）。
host / torch 侧只由 `dim` 算出 `outer/axis/inner` 传下去，**var / updates / 输出都不做 transpose**
（`op_host/index_add.asc` 与 `op_extension/index_add_torch.cpp` 均无 `MoveAxis`），
也没有 P2 那条「坐标分解必须从前到后」的坑——那条坑只属于 MoveAxis 路线。

**工作单元**：

```
单元 = (o, tTile, iTile)，按 unitsPerCore 连续切分给各核
单元内流程：
  1. 把 slab（x 上 t∈[t0,t0+tLen) × i∈[i0,i0+iLen) 这块区域）搬入 UB
  2. 按 idxChunk 分片搬入 index，标量筛出 t 落在区间内的 k（片内偏移写入 selBuf）
  3. 命中的 upd 行按 selCap 分批搬入 UB，Axpy 累加到 slab
  4. slab 整块回写 y
```

- **upd 流量 1:1（无放大）**：只搬命中的行，且每行只搬一次
- **重复索引天然正确**：同一 slab 行的多次 `Axpy` 在 V 流水内**按序执行**，
  结果等价于串行累加，**无需原子、无需去重**
- **index 扫描放大 = 单元数**：每个单元都要扫全部 index。这是本 Pattern 的主要性能风险（§5）
- **向量长度 = iLen**：`Axpy` 一次处理 `iLen` 个元素

### 2. 两条模式

| 模式 | 触发条件 | slab 形态 | 累加方式 |
|------|---------|----------|---------|
| mode0 | `inner >= 2` | `tLen × iLen` 二维块 | 向量 `Axpy` |
| mode1 | `inner == 1` | `axis` 上一段连续元素 | 逐 index 标量累加（无向量指令） |

> 分界点 `inner >= 2` 与 P5 一致：`inner == 1` 时向量指令的向量长度只有 1，向量化无意义，
> 退回标量路径反而更省（无对齐补齐、无 UB 行步长浪费）。

**mode0 的两条 UB 布局分支**：

```
availRows = pool / rowBytes        # pool = 预算 - index片与筛选列表
if (availRows >= 2)   # 整行放得下：iChunk = inner, iTiles = 1
    selCap = clamp(availRows/4, IA_MIN_SEL_CAP, IA_SEL_CAP_MAX, availRows-1)
    tChunk = availRows - selCap
else                  # 单行都放不下（inner 极大）：tChunk = 1, 行内再切 iChunk
    iChunk = align_floor(availCols / (1 + IA_MIN_SEL_CAP), align)
    selCap = clamp(availCols/iChunk - 1, 1, IA_SEL_CAP_MAX)
```

**搬运不变量（关键设计）**：

> `iTiles == 1` → `iChunk == inner`，slab = `tLen*inner` 个**连续**元素（整行整块）
> `iTiles > 1` → `tChunk == 1`，slab = `iLen` 个**连续**元素（单行内的一段）

这条不变量的作用是：**所有 `DataCopyPad` 都是 1-D 连续拷贝（`blockCount = 1`），
不使用任何 `srcStride`/`dstStride`** —— 规避步长语义与对齐歧义（步长单位、
`blockLen` 是字节还是元素在各版本间不一致，是踩坑高发区）。

- `rowStride`（UB 行步长）：`inner` 已 32B 对齐时**紧排**（= `inner`），
  此时 slab 在 GM 中连续、可整块搬运；否则补齐到 32B（逐行搬运，
  避免向量指令落在非对齐地址上）。`rowStride != inner` 时限制 `tChunk <= IA_PER_ROW_MAX_TCHUNK`
- **分段点取 32B 整数倍**，保证每段起始地址仍 32B 对齐

### 3. index 筛选：branchless + 4 路展开（性能关键）

单元内需要「扫全部 index，筛出 `t` 落在 `[t0, t0+tLen)` 的 k」。这个循环在
`tTiles` 大时占主导（成本 ∝ `tTiles × nidx`），优化收益显著（实测 2.2x）：

```cpp
// 无分支：`selp[nSel] = kk; nSel += hit;` 是经典 branchless compaction，
//   未命中时多写一次会被后续命中覆盖；selBuf 需预留展开尾巴的余量（+32B）
// 单次无符号比较判落区：t < t0 时 (t - t0) 回绕成巨值，被 rel < tLen 一并挡掉
uint32_t nSel = 0, t0u = t0, kk = 0;
for (; kk + 4 <= len; kk += 4) {
    int32_t r0 = idxp[kk], r1 = idxp[kk+1], r2 = idxp[kk+2], r3 = idxp[kk+3];
    uint32_t v0 = (uint32_t)(r0 + ((r0 < 0) ? (int32_t)axis : 0));   // 负索引修正
    uint32_t v1 = (uint32_t)(r1 + ((r1 < 0) ? (int32_t)axis : 0));
    uint32_t v2 = (uint32_t)(r2 + ((r2 < 0) ? (int32_t)axis : 0));
    uint32_t v3 = (uint32_t)(r3 + ((r3 < 0) ? (int32_t)axis : 0));
    selp[nSel] = kk;     nSel += ((v0 - t0u) < tLen) ? 1u : 0u;
    selp[nSel] = kk + 1; nSel += ((v1 - t0u) < tLen) ? 1u : 0u;
    selp[nSel] = kk + 2; nSel += ((v2 - t0u) < tLen) ? 1u : 0u;
    selp[nSel] = kk + 3; nSel += ((v3 - t0u) < tLen) ? 1u : 0u;
}
```

- **`selBuf` 容量必须是 `idxChunk * sizeof(uint32_t) + 32`**：branchless 写法在
  未命中时也会写一次，4 路展开的尾巴最多多写 3 个 uint32，`+32B` 余量足够
- 负索引修正放在**标量侧**（`r + (r<0)*axis`），因为 `CompareScalar` 只支持 half/float，
  标量比较在这里比「int 位复用 float」（P2/P5 mode1 的技巧）更省

### 4. alpha 处理

- alpha 作为 **GM 上 1 元素张量**传入，kernel 内 `alphaGm(0)` 取值（ops-nn 采纳的范式）
- **`Axpy` 的 dtype 约束（dav_c220）**：仅支持 `Tuple<T,U>` ∈
  `{<half,half>, <float,float>, <float,half>}`。Level-2 签名 `Axpy(dst, src, scalarValue, count)`
- torch 侧用**静态缓存** `GetAlphaTensor`：只在 alpha 变化时 `fill_`，
  避免每次调用都做一次 H2D/设备填充

### 5. index 扫描放大与 per-core 命中表（实测 5.19x）

**这是本 Pattern 最重要的性能机理，必须理解。**

每个单元都要**完整扫描一遍 index**。当 `tTiles` 大（UB 放不下整行 → `tChunk` 小 →
`tTiles` 大）且 `nidx` 大时，总标量迭代数 = `units × nidx`，可能远超有效计算量：

| 用例 | tChunk | tTiles | units | units×nidx | custom | torch | 对比 |
|------|--------|--------|-------|-----------|--------|-------|------|
| 4096×4096 dim0 n4096 | 10（UB 只放 10 行） | 410 | 410 | **1.7M** | 0.685ms | 0.226ms | **0.33x** |
| 4096×16 dim0 n512 | 10 | 410 | 410 | 0.21M | 0.040ms | — | 正常 |

> **诊断方法**：固定 shape、扫 `nidx`（1 → 64 → 512 → 4096），耗时若随 `nidx` 线性增长，
> 则瓶颈是 index 扫描放大而非搬运；再把 `inner` 调大（使 `tChunk` 变大、`tTiles` 变小），
> 若耗时骤降则确认。

**已实施的缓解（第一层）**：branchless + 4 路展开（§3）→ 1.549ms → 0.699ms（约 2.2x），0.11x → 0.33x。

**已实施的根治（第二层）：per-core 命中表 `ProcessUnitsHit`，实测 5.19x**

把「每个单元扫全部 index」改成「**每核扫一次 index + 各单元只扫命中表**」：

- **Phase A**：每核**扫一次** index，把落在本核 t 区间内的 `(k, t)` 写进命中表
- **Phase B**：各单元只扫命中表（实测 ~1.5 条/单元），而非全量 index
- 放大倍数从「**单元数**」降为「**核数**」，标量工作量 3.36M → ~170K

```cpp
// Phase A：每核扫一遍 index，登记命中（命中数上限 = nidx，hitCap == nidx 故不会溢出）
for (uint32_t kb = 0; kb < nidx; kb += idxChunk) {
    uint32_t len = (nidx - kb < idxChunk) ? (nidx - kb) : idxChunk;
    SyncStoM2();                       // 上一片标量读完 idxLocal 后才能覆盖
    CopyInIndex(kb, len);
    SyncM2toS();
    for (uint32_t kk = 0; kk < len; kk++) {
        int32_t r = idxp[kk];
        uint32_t v = static_cast<uint32_t>(r + ((r < 0) ? static_cast<int32_t>(axis) : 0));
        if ((v - cT0) < cTLen) { hitK[nHit] = kb + kk; hitT[nHit] = v; nHit++; }
    }
}

// Phase B：逐单元只扫命中表（批缓冲复用 selBuf：前 selCap 存 k、后 selCap 存 slab 内行号）
for (uint32_t u = uStart; u < uEnd; u++) {
    uint32_t t0 = u * tChunk, tLen = ...;
    SyncM3toM2(); SyncVtoM2();
    CopyIn1D(slabLocal, xGmBase, slabOff, tLen * inner);
    uint32_t cnt = 0;
    for (uint32_t h = 0; h < nHit; h++) {
        uint32_t rel = hitT[h] - t0;   // t < t0 时回绕成巨值，被 rel < tLen 挡掉
        if (rel < tLen) {
            batchK[cnt] = hitK[h]; batchR[cnt] = rel; cnt++;
            if (cnt == selCap) { FlushHitBatch(batchK, batchR, cnt); cnt = 0; }
        }
    }
    if (cnt > 0) { FlushHitBatch(batchK, batchR, cnt); }
    SyncVtoM3(); SyncM2toM3();         // SyncM2toM3 覆盖「本单元无任何命中（无 Axpy）」的情形
    CopyOut1D(yGmBase, slabOff, tLen * inner, slabLocal);
}
```

**启用条件**（不满足则 `hitCap = 0` 走原路径，**零回归**）：

- `mode0 && outer == 1 && iTiles == 1 && rowStride == inner && tChunk >= 2 && nidx >= 4 * tChunk`
- **必须 `outer == 1 && iTiles == 1`**：只有这样「单元号 u 即 tTile」，同一核各单元 t 区间才
  **连续**、可合并成单一区间；否则各单元 t 区间不连续，无法用一张表
- UB 预算：`fixed = idxChunk*8 + nidx*8 + 256`，tiling 侧需**用减去 hitBuf 后的预算重算
  `tChunk/selCap`**（该 shape `tChunk` 5→4）

**两个必须遵守的细节**：

1. **命中表必须按 `nidx` 条预留**（索引可重复，命中数上限就是 `nidx`）。按 t 区间跨度预留会溢出
2. **批缓冲复用 `selBuf`**（前 `selCap` 存 k、后 `selCap` 存 slab 内行号），无需新增 UB

**已知轻微代价**：`mode0 1024×4096 dim0 n256` 约 **-2%**（43.4→44.9us）。原因不是指令数
（该路径指令数更少），而是 Phase A 是**串行前导** —— 原路径的 index 扫描能与上一单元的 MTE3
写回重叠，命中表路径的表扫描太小、藏不住 MTE3。判定为可接受：该用例本身已快于 torch；
且试图用「收益比阈值」筛掉它，会把 `n1024` 这类本来有收益的用例一起筛掉
（两者理论收益比 4.17 vs 5.0，阈值刀锋）。

**仍存在的结构性限制**：命中表只覆盖 `outer == 1 && iTiles == 1` 的形态。多 outer / 多 iTile
（如 mode1 大 outer、mode0 的 `iTiles > 1`）没有等价方案；那类 shape 的代价本质是
`outer × nidx` 次标量操作，需向量化（按 k 固定、沿 o 维跨步 gather/scatter-add）才能进一步提速
（`mode1 4096×4096 dim=-1 n2048` 虽已快原生 17.5x，但绝对耗时 6.77ms 仍是最大项）。

### 6. UB 预算

```
IA_UB_BUDGET = 176KB（910B 单核 UB 192KB）    IA_SLACK = 4096
idxChunk = min(nidx, IA_MAX_IDX_CHUNK=2048) 再 8 对齐
pool = BUDGET - SLACK - idxChunk*4*2          # index 片 + 筛选列表各 idxChunk 个 int32
```

| 模式 | Buffer |
|------|--------|
| mode0 | `slabBuf = tChunk*rowStride*es`、`updBuf = selCap*rowStride*es`、`idxBuf = idxChunk*4`、`selBuf = idxChunk*4`（筛选列表，**已预留 32B 展开尾巴**） |
| mode1 | `slabBuf = tChunk*es`、`updBuf = idxChunk*es`（与 index 片同长）、`idxBuf = idxChunk*4`；**无 `selBuf`**（flat 路径不筛选，逐 index 直接标量累加） |

mode1 的 `tChunk`：`align_floor(pool / es, align)`，下限 `align`（= 32B 的元素数），
上限 `IA_MAX_COPY_BYTES / es`（受单次搬运字节上限约束）。

**同步链**：

| 事件 | 位置 | 作用 |
|------|------|------|
| `MTE2_S` | mode0：slab + index 片搬入 → 标量筛 index；mode1：slab + index/upd 片搬入 → 标量累加 | slab 到位才能作为累加目标 |
| `MTE2_V` | upd 行搬入 → `Axpy`（仅 mode0） | |
| `S_MTE2` | **每片 index/upd 搬入前** | 上一片标量/向量未读完就覆盖会稳定错位 |
| `V_MTE3` | `Axpy` 完成 → 整块回写（仅 mode0） | |
| `S_MTE3` | mode1 标量累加写完 slab → 整块回写 | mode1 无向量指令，用标量事件替代 `V_MTE3` |
| `MTE2_MTE3` | slab 搬入 → 回写（本单元无任何命中、未做 Axpy 时） | |
| `MTE3_MTE2` | 回写 → 下一单元 slab 搬入 | |

## Tiling 字段

| 字段 | 含义 |
|------|------|
| mode | 0：向量 slab（inner >= 2）；1：flat 标量（inner == 1） |
| outer / axis / inner / nidx | 语义分解参数 |
| tChunk / tTiles | slab 覆盖的 t 行数 / `ceil(axis / tChunk)` |
| iChunk / iTiles | slab 覆盖的 i 列数（`iTiles==1` 时 == inner） |
| rowStride | UB 内 slab/upd 行步长（元素，32B 对齐） |
| units / unitsPerCore / tailUnitsLastCore | `outer*tTiles*iTiles` / 每核单元数 / 尾核单元数 |
| idxChunk / selCap | 每次搬入的 index 元素数 / 每批并行搬入的 upd 行数 |

**常量**：`IA_MAX_COPY_BYTES=65535`、`IA_UB_BUDGET=176KB`、`IA_ALIGN=8`、
`IA_MAX_IDX_CHUNK=2048`、`IA_SEL_CAP_MAX=16`、`IA_MIN_SEL_CAP=4`、`IA_PER_ROW_MAX_TCHUNK=2048`

> **`DataCopyExtParams::blockLen` 是 `uint16_t`**（不是 uint32_t，CANN 9.0.0 实测）：
> 单次搬运 ≤ 65535 字节，超了报 `-Wc++11-narrowing` 编译错。
> 代码统一 `static_cast<uint16_t>(...)`，片长由 host 侧按 `IA_MAX_COPY_BYTES` 约束。

## 已验证场景（910B / CANN 9.0.0）

- 直调通路 **23/23 全通过**
- PyTorch 通路 **60/60 全通过**：2D/3D × dim 0/1/-1 × FP32/FP16 ×
  负索引 × 重复索引 × alpha ∈ {0.5, 1.0, 2.0, -1.5} × 大 nidx 专项

**性能（vs 原生 `torch.index_add`）**：**1.04x ~ 17.50x，全部 ≥ 1.0，无短板** ——
此前 `mode0 4096×4096 dim0 n4096` 的 0.24x 已由 §5 的 per-core 命中表消除。

| 用例 | custom(ms) | torch(ms) | 加速比 |
|------|-----------|-----------|--------|
| mode0 1024×4096 dim0 n256 | 0.0402 | 0.1064 | 2.65x |
| mode0 1024×4096 dim0 n1024 | 0.0402 | 0.1066 | 2.65x |
| mode0 4096×4096 dim0 n4096 | 0.1370 | 0.1644 | **1.20x** |
| mode0 64×4096×256 dim1 n64 | 0.4437 | 1.8390 | 4.14x |
| mode1 4096×4096 dim=-1 n2048 | 6.7653 | 118.3628 | 17.50x |
| mode1 4096×4096 dim=-1 n4096 | 13.5068 | 233.6441 | 17.30x |
| mode1 4×8192 dim=-1 n4096 | 0.1375 | 0.3213 | 2.34x |
| mode1 2×20000 dim=-1 n10000 | 0.3088 | 0.3202 | 1.04x |
| mode1 2×131072 dim=-1 n4096 | 0.0859 | 0.2163 | 2.52x |
| mode1 64×256 dim=-1 n8192 | 0.5388 | 8.5884 | 15.94x |

（测量口径：wall-clock、`iters=50, warmup=20, rounds=5` 取最小值；
命中表 A/B 结论用**同进程交错 + rounds=25** 取得。）

- `ok` 列必须用 `torch.allclose(atol=1e-5, rtol=1e-5)` 而非 `torch.equal`：
  fp32 下累加顺序与 native 不同，bitwise 相等不成立（用 `equal` 会得到「全 False」的误报）
- fp16 累加是**逐次舍入**（每次 `Axpy` 后回落 fp16），与 golden 的
  「fp32 累加再一次回落」固有差异约 1~2 ULP（`nidx` 大时实测 max diff ~0.0078）→
  容差取 `1e-2`

## torch 接入层：`stream(true)` 的语义（🔴 必读，踩坑最深）

PyTorch 通路首调用随机错（60 用例中 36~48 通过、失败集每次不同、失败时
`max|y-x| = 0.0` 即 kernel 完全没施加更新）的**真正根因**：

```cpp
// torch_npu/csrc/core/npu/NPUStream.cpp
NPUStream::stream(const bool need_empty) {          // L690
    if (!need_empty) { return cur_ptr->stream; }
    return stream();                                 // L364
}
NPUStream NPUStream::stream() {                      // L364
    if (GetPerStreamQueue() && repo->CheckInit()) {
        cur_ptr->repo->MakeSureQueueEmpty();         // ← 清空 torch_npu 延迟任务队列
    }
    ...
}
```

> **`stream(true)` 会 `MakeSureQueueEmpty()`，把此前 enqueue 的 torch 算子真正下发。**
> 因此 **`stream(true)` 必须在「所有 torch 侧算子之后」调用**！
> 顺序颠倒（先取 stream、再执行 `fill_`）会让 `fill_` 留在队列里没下发，
> kernel 读到 `at::empty` 的未初始化内存 → 表现为「首调用随机错、重跑即好」。

正确顺序：

```cpp
// 1. 先完成所有 torch 侧动作（tiling H2D、alpha fill_）
at::Tensor& tilingTensor = GetTilingBuffer();
SyncTilingToDevice(tilingTensor, tiling);
at::Tensor& alphaTensor = GetAlphaTensor(out.options(), alpha);
// 2. 再取 stream（MakeSureQueueEmpty 才覆盖得到上面这些算子）
auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
// 3. 最后 launch
index_add_kernel(tiling.blockNum, nullptr, aclStream, xPtr, idxPtr, updPtr, alphaPtr, yPtr, tPtr);
```

**诊断特征（可用于识别同类问题）**：

| 观察 | 结论 |
|------|------|
| 失败集每次不同 | 不是数据相关确定性错误 |
| 对失败用例重复 5 次 → 恒为「第 1 次失败、后 4 次全过」 | 首调用读到未初始化内存 |
| fp16 首调用出现 `nan` | 同上（未初始化内存的典型表现） |
| 失败集合与「alpha/dtype 自上一用例以来发生变化」一一对应 | 指向 `fill_` 未下发 |
| 加 `torch.npu.synchronize()` 或先 warmup 即规避 | 同上 |

## 附：排序合并重复索引（Sort32，思路性方案）

> ⚠️ **本节为早期原型思路，未经上板验证，不可作为选型依据。**
> 其中所有阈值（`MAX_SORT_SIZE = 8192`、重复率 > 50%）均为**启发式口径**，非本 Pattern 的模式判据。
> 现有的 `index_add_custom` 工程**未采用**该方案（采用的是 §核心思路 的「单元区间不重叠 + slab 一次回写」）。
> 保留本节仅为记录思路。

**动机**：当多个 index 命中同一目标行时，直觉做法是原子加/多次写；排序路线改为
**先排序分组、组内在 UB 累加、每组只写 GM 一次**。

### 冲突类型

| 冲突类型 | 算子 | 问题 | 优化方向 |
|---------|------|------|---------|
| 读冲突 | index_select / gather | 多个索引读同一位置 | 排序合并读取，减少 GM 读 |
| **写冲突** | **index_add / scatter_add** | 多个索引写同一位置 | 排序合并写入，减少写次数 |

```
index = [0, 0, 1]
朴素：self[0] += src[0]；self[0] += src[1]（两次写同一位置）
排序：UB 内算 src[0] + src[1]，一次写入 self[0]
```

### 三步实现

**Step 1 排序**：`int64` index 需先 `Cast` 成 `float`（Sort 的 value 只支持 `float`/`half`），
再用 `Sort32` 同时排序「值」与「原始位置」：

```cpp
Cast(sortValue, indexLocal, RoundMode::CAST_NONE, indexSize_);
for (int64_t i = 0; i < indexSize_; i++) { sortIndex.SetValue(i, static_cast<uint32_t>(i)); }
int32_t repeatTime = (indexSize_ + 15) / 16;
Sort32(sortValue, sortValue, sortIndex, repeatTime);   // sortValue=排序后目标索引, sortIndex=原始位置
```

**Step 2 分组**：`sortValue` 已有序，线性扫描取相同目标索引的连续区间 `[groupStart, groupEnd)`。

**Step 3 组内累加后一次写回**：读 `self` 原值 → `Duplicate` 清零累加器 →
组内逐个 `DataCopyPad(src) → Muls(alpha) → Add` → 加 `self` 原值 → 一次 `DataCopy` 写回（**无需原子**）。

### Sort32 API 关键点

| 项 | 说明 |
|----|------|
| 签名 | `Sort32(dst, src0, src1, repeatTime)`；`src0` 待排序值、`src1` 伴随索引 |
| `repeatTime` | `(count + 15) / 16`（每次处理 16 个元素） |
| Value 类型 | 只支持 `float` / `half` → `int64` / `int32` 索引必须先 `Cast` |
| Index 类型 | 只支持 `uint32_t` |

### 适用场景判断（启发式，未验证）

| 场景 | 推荐 | 原因 |
|------|------|------|
| 索引数量 ≤ `MAX_SORT_SIZE` = 8192 | 排序优化 | 排序开销可接受 |
| 索引数量 > 8192 | 不启用 | 排序开销大、UB 空间不足 |
| 无重复索引 | 直接处理 | 无收益 |
| 重复率 > 50% | 排序优化 | 大幅减少 GM 写次数 |

**与本 Pattern 已交付路线的关系**：已交付路线用「单元 `(t 区间, i 区间)` 互不重叠」
从根上消除了写冲突（同一 slab 行无论被几个 index 命中，都只在本单元内累加、只回写一次），
因此排序路线在**写次数**上并无额外收益；其潜在价值需在「重复率极高且 index 可整批驻留 UB」
时另行上板验证。

## 与相邻 Pattern 的区分

- **P5（IndexSelect）**：同为「沿轴 + 1-D index」，但 IndexSelect 是**纯读**（输出连续、
  可向量化搬出），IndexAdd 是**读-改-写**（需 slab RMW、需处理多 index 命中同一行）
- **P3/P4（Scatter 系）**：index 形状与输出一致（逐元素映射），本 Pattern 的 index 是
  沿轴 1-D（一行 index 影响整个 `inner` 块），且**输出元素单属主**是靠
  `(t 区间, i 区间)` 不重叠保证的，与 scatter 的「输出元素空间 64B 整块分核」思路同源但实现不同
- **不要给 IndexAdd 用 GM 原子加**：`(o*axis+t)*inner` 不保证 32B 对齐（§「核心约束」）
