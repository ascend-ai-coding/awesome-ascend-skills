# P7: 沿轴重复展开（RepeatInterleave）

> 状态：已补充（2026-09-24，基于 RepeatInterleave 算子 910B（dav_c220）实战验证）
> 硬件基线：Ascend 910B3 / dav_c220 / 40 个 AIV core / 单核 UB 192KB / L2 192MB / CANN 9.0.0
>
> **精度状态：bit-exact（max diff = 0）—— 这是结构性保证，不是调出来的**
> **性能状态：大 shape 稳定领先（1.29~2.31x），`dim=None`（flatten）34~36x；
> 小 shape 受 host 固定开销支配，两轮结果会翻转**

## 适用场景

```
out = x.index_select(dim, arange(axis).repeat_interleave(repeats))
```

维度分解（与 P5/P6 同构）：

- `outer = prod(shape[:dim])`
- `axis = shape[dim]`
- `inner = prod(shape[dim+1:])`
- `total = sum(repeats)`

元素映射：

```
x[(o * axis + t) * inner + i]  ->  y[(o * total + k) * inner + i]
```

即把 `dim` 轴上的第 `t` 个元素重复 `repeats[t]` 次后依次铺开，其余维度不变。

- 典型算子：`torch.repeat_interleave`（`repeats` 可为标量或 1-D tensor）
- **与 P5（IndexSelect）的区别**：IndexSelect 的 index 是外来的、任意的；
  RepeatInterleave 的「index」是 `arange(axis).repeat_interleave(repeats)`——**有序且可解析**，
  因此**不物化 index**（见下）

**实现路线（路线 B：直接按 repeats 连续写）**：不物化 `srcIndex`，kernel 直接按
「第 `t` 个输入元素占据输出区间 `[prefix[t], prefix[t]+r[t])`」推进输出行号。
无 Cast、无浮点算术 → **精度 bit-exact 是结构性保证**。

## 模式总览：两档动态策略

| 模式 | 触发条件 | 策略 | UB buffer |
|------|---------|------|-----------|
| mode0 | `inner ≥ 2` | **段拷贝**：纯 MTE2 搬入 + 标量寻址 + MTE3 搬出，**无输出 buffer、无 `PIPE_V`** | `xBuf` + `tBuf` |
| mode1 | `inner == 1` | **UB 内构造展开下标 + `Gather`**（与 gather / index_select 的 mode1 同构） | `xBuf` + `idxBuf` + `yBuf` + `tBuf` |

> 分界点与 P5 一致：`inner == 1` 时连续搬运粒度只有 1 个元素，搬运动词退化成逐行标量，
> 改用「UB 内构造展开下标 + 向量 `Gather`」一次处理整块。

## 核心思路

### 1. 两条路线，选「不物化 index」

- **路线 A（物化 srcIndex）**：先算出 `arange(axis).repeat_interleave(repeats)` 存成 index 张量，
  再走 index_select 的路子 —— 多一次全量 index 的构建与读写。
- **路线 B（按 repeats 连续写）**：直接用前缀和推进输出行号，**不物化 index**。
  本实现采用路线 B，也因此**没有任何浮点运算**，bit-exact 是结构性的。

### 2. mode0 实现要点（段拷贝，主力路径）

搬运单位是 **`(t, s)`**：`t` 为 `axis` 维下标，`s` 为 `inner` 维段下标（共 `segPerRow` 段）。

1. **UB 只放「一批源行的同一段」**：`batchT * rowStride * es`，**与 `inner` 解耦**。
   这是关键设计 —— 若按「一批源行的全部段」开 UB，当**单行段总字节 > UB 预算**时
   `batchT` 会被钳到 1 仍然越界（WB-10 缺陷）。逐段循环复用则天然安全：
   单段字节 `rowStride * es ≤ 65536 < avail`，`batchT` 天然 ≥ 2，**不存在 `B == 0` 越界路径**。
2. **MTE2 只搬一次源行批次**：一批 `batchT` 个源行搬入后，MTE3 依次写出 `r[t]` 个
   **连续输出行**的对应段。
3. **输出行号用 `k` 累加器推进**（`k += r`），**禁止逐行做 `k / r` 整数除法**。
   各段起点均为同一个 `kBase`，段处理完返回 `kBase + Σr`（各段一致）。
4. **`r == 0` 直接 `continue`**，不产生任何 store。
5. **同步红线**：每段独立 `load → SyncM2toM3 → store → SyncM3toM2`。
   `SyncM2toM3` 等本段 x 搬入完成；`SyncM3toM2` 在下一段 load 覆盖 `xBuf` 前等本段 store 读完。

```cpp
for (uint32_t o = oStart; o < oEnd; ++o) {
    uint32_t k = kStart;                       // 输出行累加器（无整数除法）
    for (uint32_t tb = tStart; tb < tEnd; tb += batchT) {
        uint32_t nT = min(tEnd - tb, batchT);
        LoadBatchRepeats(tb, nT);
        uint32_t kBase = k;
        for (uint32_t s = 0; s < segPerRow; ++s) {
            LoadBatchSegments(o, tb, nT, s);   // 逐行发起第 s 段搬入
            SyncM2toM3();                      // 等本段 x 搬入完成（红线）
            k = StoreBatchSegments(o, kBase, nT, s);  // 返回 kBase + Σr
            SyncM3toM2();                      // 下一段 load 覆盖前，等本段 store 读完（红线）
        }
    }
}
```

> **穿刺结论（不要重走）**：曾尝试用 `DataCopyPad` 多块复制写（`blockCount = r`、`srcStride = 0`）
> 替代逐行搬出。**`srcStride` 的语义是「相邻数据块的间隔」，`srcStride = 0` 表示 UB 侧块间连续，
> 并不等于「重复读同一 UB 段」** —— 因此多块复制写**不能**替代逐行搬出
> （O5 穿刺失败，保留开关 `RI_USE_MULTIBLOCK_STORE` 默认 0）。

### 3. mode1 实现要点（展开下标 + Gather）

1. **x 块按 `t` 贪心分块**：从当前 `t` 起向后累加 `r`，直到 `K = Σr > kChunk` 为止；
   块内 `t` 数 `L ≤ kChunk`。**单个 `t` 的 `r > kChunk`** 时拆成 `ceil(r / kChunk)`
   次同值展开（每次 `chunk = min(remaining, kChunk)`）。
2. **索引块每块构建一次、块内跨 `o` 复用**（O3）：`idxBuf` 存的是**展开后的源字节偏移**
   （局部 `t` 下标 × `es`），与 `o` 无关，因此一个块内循环所有 `o` 时**只构建一次**。
3. **Gather 后整块一次搬出**（O4）：`Gather(yLocal, xLocal, offsetLocal, 0, alignUp(K))`，
   再一次 `DataCopyPad` 写 `K` 个元素。
4. **`BuildIndexBlock` 两趟（满足 UB 向量访问 32B 对齐硬约束）**：
   - **Pass 1（标量）**：写各 run 的**对齐头部**（每 run 最多 7 个元素）+ 尾部补 0；
   - **Pass 2（向量）**：用 `Duplicate` 写各 run 的**对齐主体**（与 Pass 1 的头部区间不相交）。
   - 标量步数是 `O(L)`，**不随 `outer` 增长** —— 这是该设计能扩展到大 `outer` 的关键。
5. **同步链**：`SyncVtoS`（上一块 Gather 读完 `idxBuf` 后才能被标量覆盖）→
   `SyncStoV`（标量写 `idxBuf` → 向量读）→ 每个 `o`：`SyncVtoM2` / `SyncM2toV`（Gather 前等 x）/
   `SyncM3toV`（上一个 `o` 的 MTE3 读完 `yBuf`）/ `SyncVtoM3`（`yBuf` 就绪）。

### 4. 分核：前缀和分核表（`tParts × oParts` 二维）

因为每行展开后的长度可能不同（tensor repeats），分核**不能按行均分**，
而要按**展开后的输出元素空间**均衡切分。

```cpp
uint32_t Q = (core < outer) ? core : outer;        // o 维分片数
if (Q == 0) Q = 1;
uint32_t P = core / Q;                             // t 维分片数
if (P < 1) P = 1;
if (P > RI_MAX_T_PARTS) P = RI_MAX_T_PARTS;
t.tParts = P;  t.oParts = Q;  t.blockNum = P * Q;
t.oPerCore = (outer + Q - 1) / Q;
// kernel：core c → p = c / oParts（t 分片），q = c % oParts（o 分片，oStart = q * oPerCore）
```

**分核表 `coreTable`**（内联进 tiling，`2*(RI_MAX_T_PARTS+1)` 个 uint32）：
`coreTable[2p] = tStart_p`、`coreTable[2p+1] = kStart_p`（`p ∈ [0, P]`，单调非降），
末尾固定 `coreTable[2P] = axis`、`coreTable[2P+1] = total`。

| 形态 | 分核表求法 |
|------|-----------|
| **int repeats** | **解析式**：`tStart_p = axis * p / P`，`kStart_p = r * tStart_p` |
| **tensor repeats** | `prefix` 单调非降 → **单次线性扫描**（`O(axis)`）找首个 `prefix[tp] >= total * p / P` 的 `tp` |

> **收尾**：张量形态强制把末项置为 `(axis, total)` —— 尾部的 `r == 0` 元素不产生输出，
> 覆盖它们无副作用。

### 5. 关键常量（`repeat_interleave_tiling.h`）

```cpp
constexpr uint32_t RI_MAX_COPY_BYTES   = 65535;      // DataCopyPad blockLen 上限（字节，不放开到 2MB）
constexpr uint32_t RI_UB_BUDGET        = 176 * 1024; // 910B 单核 UB 192KB，取 176KB 留余量
constexpr uint32_t RI_ALIGN            = 8;          // 向量 count / int32 8 对齐（32B 块 = 8 个 int32）
constexpr uint32_t RI_ALIGN_BYTES      = 32;         // UB 地址 32B 对齐硬约束
constexpr uint32_t RI_MAX_BLOCK_COUNT  = 4095;       // DataCopyExtParams::blockCount 上限
constexpr uint32_t RI_MODE0_MAX_BATCH  = 4096;       // mode0 每批源行数上限
constexpr uint32_t RI_MAX_K_CHUNK      = 8192;       // mode1 单块输出元素上限
constexpr uint32_t RI_MODE1_MIN_CHUNK  = 64;         // mode1 单块元素下限（64 对齐）
constexpr uint32_t RI_MAX_T_PARTS      = 64;         // 分核表最大 t 分片数（>= 最大核数）
```

### 6. 双接口

```cpp
// 标量 repeats
npu::repeat_interleave_int_custom(at::Tensor input, int64_t repeats, int64_t? dim = None);
// 张量 repeats（逐元素不同重复次数）
npu::repeat_interleave_tensor_custom(at::Tensor input, at::Tensor repeats, int64_t? dim = None);
```

- `dim = None` 时先把输入展平为 1-D（`axis = numel`）再展开（对应 PyTorch 的 flatten 语义）。
- 张量形态：`repeats` 必须 **1-D**、`numel == axis`、**int32/int64**（int64 在 host 侧转 int32）。
- 标量形态：kernel 的 `repeats` 指针传 `xPtr` **占位**（kernel 不解引用，`repeatsInt` 常量生效）。

### 7. host 侧实现要点（固定开销相关）

1. **`stream(true)` 必须在所有 torch 侧算子之后**：`contiguous` / `to` 都必须在取 stream 之前；
   tensor 形态要在 `stream(true)` **之后**才做 `repeats` 的 D2H 读取，才能读到已算完的数据。
2. **核数查询 `static` 缓存**：`aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 的结果缓存，
   避免每次 launch 重复查询。
3. **tiling device buffer 静态复用 + 内容比较跳过 H2D**：固定分配一次；
   `memcmp` 比对内容，**仅当 tiling 变化时才发 H2D**（H2D memcpy 约 **40us**，
   是固定开销主要来源；shape/repeats 稳定的推理循环里可完全省掉）。
4. **退化输入提前返回**：`numel == 0 || total == 0` 时直接返回空输出，**跳过 launch**（避免空指针）。
5. **输出用 `at::empty`**：纯读型算子不需要初值，不要用 `zeros_like`
   （多一次 kernel 且顺序语义易踩坑）。
6. **`total` 上界校验**：`total <= UINT32_MAX`（tiling 中 `total` 为 uint32）。

### 8. 踩坑（实测）

- **UB 越界 → 错误码 507035**：mode1 构造展开下标时，分片长度须受 `RI_MAX_K_CHUNK` /
  `RI_MODE1_MIN_CHUNK` 约束；下标数组长度按 `total` 而非 `axis` 估算，否则 UB 溢出。
- **输出行累加器未回传**：mode0 逐行累加输出偏移时，行内累加器必须在换行时回写/重置，
  否则后续行的输出位置错位。
- **UB 按「一批源行的全部段」开会越界（WB-10）**：单行段总字节 > UB 预算时 `batchT` 被钳到 1
  仍越界。正解是**逐段循环复用 `xBuf`**（`batchT * rowStride * es`，与 `inner` 解耦）。
- **多块复制写不能替代逐行搬出（O5 穿刺失败）**：`DataCopyParams::srcStride` 是「相邻数据块的
  间隔」，`srcStride = 0` 表示 UB 侧块间**连续**，**不是**「重复读同一 UB 段」。
- **`idxBuf` 需额外对齐余量**：`BuildIndexBlock` 按 8 对齐写 `Duplicate` 主体，
  容量取 `RiCeilAlign(kChunk, 8) + 8`，否则对齐写入越界。

## Tiling 字段

| 字段 | 含义 |
|------|------|
| mode | 0：`inner ≥ 2` 段拷贝；1：`inner == 1` 展开下标 + Gather |
| outer / axis / inner / total | 语义分解参数与展开后轴长 |
| tParts / oParts / oPerCore / blockNum | `t × o` 二维分核（blockNum = P × Q） |
| coreTable[] | 分核表：`[2p]=tStart_p`、`[2p+1]=kStart_p`，末项固定为 `(axis, total)` |
| batchT | mode0 每批源行数 |
| segPerRow / rowStride | `inner` 分段数 / UB 行步长（32B 对齐） |
| kChunk | mode1 单块展开元素数上限（≤ `RI_MAX_K_CHUNK`） |
| repeatsIsTensor / repeatsInt | 张量 repeats 标志 / 标量 repeats 值 |

## 已验证场景（910B / CANN 9.0.0）

8 条用例 **bit-exact 全通过**（max diff = 0），覆盖标量 repeats / tensor repeats /
`dim ∈ {0, 1, -1, None}` / `inner ≥ 2` 与 `inner == 1` 两档。

**性能（vs 原生 `torch.repeat_interleave`）**：wall-clock、`iters=50, warmup=20, rounds=5`
取 min-of-5，每轮前 `torch.npu.synchronize()`。两轮独立 bench：

| 用例 | 第 1 轮 custom/torch | 对比 | 第 2 轮 custom/torch | 对比 | loc |
|------|---------------------|------|---------------------|------|-----|
| inner≥2 1024x4096 d0 r=2 | 0.0429 / 0.0251 | 0.59x | 0.0435 / 0.0400 | 0.92x | L2 |
| inner≥2 4096x4096 d0 r=2 | 0.0527 / 0.0680 | 1.29x | 0.0626 / 0.0816 | 1.30x | HBM |
| inner==1 1024x4096 d-1 r=2 | 0.1997 / 0.1593 | 0.80x | 0.2006 / 0.1646 | 0.82x | L2 |
| inner==1 4096x4096 d-1 r=2 | 0.2967 / 0.5937 | 2.00x | 0.3034 / 0.7014 | 2.31x | HBM |
| mid-axis 8x32x16 d1 r=2 | 0.0448 / 0.0247 | 0.55x | 0.0458 / 0.0484 | 1.06x | L2 |
| flatten 8x32x16 dNone r=2 | 0.0429 / 1.4827 | **34.54x** | 0.0417 / 1.5097 | **36.24x** | L2 |
| tensor 4096x4096 d0 r[4096] | 0.2171 / 0.4659 | 2.15x | 0.2301 / 0.4602 | 2.00x | HBM |
| tensor 4096x4096 d0 mixed0 | 0.2078 / 0.3929 | 1.89x | 0.2125 / 2.8824 | 13.57x | HBM |

（对比 = torch 耗时 / custom 耗时，>1 表示本实现更快。）

**结论**：

- **大 shape 稳定领先**：`inner≥2 4096² d0` 1.29~1.30x；`inner==1 4096² d-1` 2.00~2.31x；
  tensor repeats `r[4096]` 2.00~2.15x。
- **`dim=None`（flatten）压倒性优势 34~36x**：原生 torch 在 flatten 路径退化严重
  （1.48ms vs 0.042ms）。
- **小 shape 在波动区间**：`1024x4096 d0` 0.59→0.92x、`mid-axis 8x32x16` 0.55→1.06x，
  **两轮翻转说明这些用例被 host 固定开销 + 设备波动支配**，绝对差距 ≤ 0.02ms，
  不是算法问题（参见 [index_select.md](index_select.md) §Host 固定开销与输出缓冲池）。
- **`tensor mixed0` 第 2 轮 13.57x 是噪声**：该用例两轮离散度分别 14.3% / **729.5%**，
  原生耗时 0.39ms ↔ 2.88ms 剧烈漂移，**不可作为结论**（按测量纪律需复采）。

> **口径提醒**：`loc = L2` 表示流量 < L2 容量（192MB），测得的是 **L2 带宽**而非 HBM 带宽。

## 与相邻 Pattern 的区分

- **P5（IndexSelect）**：index 由外部给定、任意无序，必须物化并搬入；
  RepeatInterleave 的展开位置**由 repeats 解析确定**，可不物化 index，因此 bit-exact。
- **P2（Gather）**：index 与输出同形、每行独立，需要坐标分解；
  RepeatInterleave 的 `outer/axis/inner` 分解与 P5/P6 一致，**不需要 transpose**。
- **P1（扁平索引取数）**：`dim = None` 时语义等价于在展平后的 1-D 张量上展开，
  但实现上仍走本 Pattern 的 mode0/mode1（不物化 index），不走 P1 的扁平取数路径。
