# P3/P4: 写索引（Scatter 系）

> 状态：已补充（2026-09-22，基于 Scatter / ScatterAdd 算子 910B（dav_c220）实战验证，
> 覆盖 P3 覆盖写与 P4 累加；两者共用同一分核与核内流程，仅写出语义不同）
> 硬件基线：Ascend 910B3 / dav_c220 / 40 个 AIV core / 单核 UB 192KB / CANN 9.0.0
>
> **性能状态：已优化，全面反超原生 torch**（UB 内 RMW + 批量回写，
> 覆盖写 3.28~3.43x / 累加 1.95~2.76x；优化前仅 0.39~0.71x）

## 适用场景

- 沿指定轴 `dim` 按 `index` 把 `src` 写到输出：`out[pre..., index[pre..., j]] = src[pre..., j]`（覆盖写）
- 累加语义：`out[pre..., index[pre..., j]] += src[pre..., j]`
- `index` 与 `src` 形状相同；`index` 的形状除 `dim` 维外必须 ≤ 输入对应维
- 典型算子：ScatterUpdate / ScatterElements（P3）、ScatterAdd / ScatterReduce（P4）
- **输出无需调用方预填**：kernel 从**输入**整片读 RMW 基准、写独立输出，未命中的位置与
  「无对应 index 行」的输出行都按输入原值写入，所以 host 侧用 `empty_like` 分配即可，
  **不要用 `clone` 给输出做初值**（省一次与输出等量的设备拷贝；且 clone 进 torch_npu
  延迟任务队列后对直调 kernel 不可见）。详见 §4

## 核心思路

### 1. 轴归一化：只实现尾轴，非尾轴由 host 侧 transpose

- Kernel 只实现**尾轴 Scatter**，非尾轴场景在 host / torch 侧先 `MoveAxis` 把 scatter 维
  transpose 到尾轴（输入 + index + src 一起转），跑完 kernel 再 transpose 回原形状
- transpose 用通用 `MoveAxis`（线性索引 ↔ 多维坐标分解）
  > **坑**：坐标分解必须**从前到后**循环（`for i = 0..n-1`）。从后向前分解会把整个
  > 线性索引吃掉在第一维上，转置后数据全错，大 shape 还会段错误。

  把 `dim` 换到最后一维的 permute 构造（`self` / `index` / `src` 三者用同一 perm）：

  ```cpp
  bool needTranspose = (dim != ndim - 1);
  if (needTranspose) {
      // 交换 dim 和 ndim-1；例：[M, N] dim=0 → permute [1,0] → [N, M]
      perm.resize(ndim);
      for (int64_t i = 0; i < ndim; i++) {
          if (i < dim) perm[i] = i;
          else if (i == dim) perm[i] = ndim - 1;
          else if (i < ndim - 1) perm[i] = i;
          else perm[i] = dim;
      }
      selfT  = self.permute(perm).contiguous();
      idxT   = index.permute(perm).contiguous();
      srcT   = src.permute(perm).contiguous();
  }
  ```

  转置后语义统一为 `selfT[o][index[o][k]] = srcT[o][k]`（`k = 0..indexAxisSize-1`）。

- **dim=0 是 NPU 平台已知问题**：PyTorch 内置 `scatter_` 在 NPU 上 `dim=0` 会触发
  aicore 异常（error 507015）。自定义算子靠上面的 transpose 策略正常支持 `dim=0`。

### 2. 硬件前提：dav_c220 无 Scatter 指令，只能标量散射 —— 但要在 UB 里散写

`dav_c220` 上 UB 版 `Scatter` 指令为 `ASCENDC_REPORT_NOT_SUPPORT`，**不能**像 Gather 那样
用硬件指令在 UB 内完成散射；且散射地址由 index 值决定、不可预测，**无法用连续 `DataCopy`
搬出**。所以标量写是唯一可行路径。

> **核心结论（本 Pattern 性能的转折点）**：标量散射写是不可避免的，但**写的目标可以是
> UB 而不是 GM**。AI Core 标量写 GM 的延迟极高（实测约 55 cycles/迭代），而标量写 UB
> 快得多。因此改为与原生 `ops-nn scatter_elements_v2` 一致的「**UB 内读改写（RMW）+
> 批量回写**」：

```
DataCopyPad 从**输入**整片读入基准片（RMW 基准 = 输入，不是输出）
  → index / src 分片搬入 UB
  → 向量化「负索引修正 + 片基址重定位」
  → 标量在 UB 内散写（8 路展开、裸指针 __ubuf__）
  → DataCopyPad 整片回写输出
```

注意基准片取自**输入**而非输出：输出与输入形状、布局一致且偏移一一对应，
未命中位置本应保留输入原值，因此直接读输入片做读改写、写独立输出即可，
**不需要调用方先把输入拷到输出**（去掉了 `clone` 这一步全量拷贝）。

这一改动把「4.2M 次 GM 标量写」变成「UB 标量写 + 与输出等量的批量搬运」，
是本算子从 0.45x 翻转到 3.2x 的主因。

### 3. 分核策略：按「输出元素空间」64B 对齐整块分核（核心，含硬件红线）

> **这是本 Pattern 最重要的结论，也是踩坑最深的地方。**

**错误做法（反模式）**：按 index 元素总量均分给各核（每核处理 `numel(index)/core` 个元素）。
在 Gather 上这是对的（输出连续、与 index 一一对应），但**在 Scatter 上会大面积丢数据**。

**实测现象**：`reduce=0` 时输出中大量元素未被写入（`2×30000` 丢失 62%）。排查特征：

| 排查手段 | 结果 | 结论 |
|---------|------|------|
| 输出 memset(0) / 不初始化 | 仍丢 | 不是初始化问题 |
| 下标哨兵 `y[e] = e + 0.5` | 仍丢且 `wrong=0` | **只有缺失、没有错值**（值不是写错位置） |
| 改用 MTE3（`Duplicate` + `DataCopyPad` 连续写） | 4096/4096 全写 | 连续搬出无问题，问题在标量散射写 |
| `PipeBarrier<PIPE_ALL>` | 无效 | 不是核内同步问题 |
| `DataCacheCleanAndInvalid`（各 DcciDst） | 无效 | 不是 dcache 未刷问题 |
| 尺寸扫描 | 阈值在 1000~2000 元素/核之间 | 与每核负载规模相关 |
| 丢失形态 | 恒为每核 head/tail **恰好 8 个连续 fp32（= 32B）** | 粒度指向 **32B sector / 64B cache line** |
| **改为 16 元素（64B）对齐分核** | **所有 shape 丢失为 0** | 根因确认 |

**根因**：AI Core 标量（S 流水）写 GM 时，**两个核并发写同一条 64B cache line 的不同 32B
sector，会丢失其中一个 sector**。原「按 index 元素量分核」在真实 scatter 下各核的输出地址
范围大量交错 → 大量元素落在与其他核共享的 cache line 上 → 62% 丢失。

**修复方案**：按**输出元素空间**分核，把展平输出 `[0, xTotal)` 切成 64B 对齐的等长块
（`outBlockSize = 64 / sizeof(T)`：fp32 为 16 元素、fp16 为 32 元素），每核独占**一段连续整块**
（工作单元 = 该核的输出段）：

```
totalBlocks = ceil(xTotal / outBlockSize)
base = totalBlocks / blockNum;  rem = totalBlocks % blockNum
bStart = blockIdx * base + min(blockIdx, rem)
bEnd   = bStart + base + (blockIdx < rem ? 1 : 0)
gStart = min(bStart * outBlockSize, xTotal)
gEnd   = min(bEnd   * outBlockSize, xTotal)
```

这样任意两个核写到的 cache line 都不重叠，且**同一输出元素只由一个核负责**。

> **红线 1**：分核基准必须是**输出元素空间**且按 64B 对齐整块切分。
> 「按 index 元素空间对齐」无效——index 与输出在真实 scatter 下不是线性对应关系
> （每行的 index 值是随机的），对齐 index 无法保证输出 cache line 不重叠。
> **红线 2**：分核总量必须用**输出元素总数 `xTotal`**，不能用 `rows * xAxisSize`。
> 当 index 前部维小于输入对应维时（如 `x=(8,32,16)` / `index=(4,16,8)` / `dim=1`），
> `rows * xAxisSize = 32*32 = 1024` 而输出实际有 `8*16*32 = 4096` 元素，
> 会有一半以上输出被排除在分核范围外（实测 Max diff 3.95）。

**P4 累加的额外收益**：由于同一输出元素只由一个核写，`reduce=1` 的累加天然确定、
**无需任何原子操作**，也不需要排序/去重/聚合（与 950PR 上推荐 SIMT+atomic 的结论不同，
此处分核本身已消除跨核冲突）。走 UB RMW 后这条结论依然成立——每个输出片只被一个核
读改写，不存在跨核 read-modify-write 竞态。

### 4. 核内处理：遍历**输出全部行** → 映射到 index 行 → 逐行逐片 RMW

分核基准是输出空间，而数据（index/src）是按**行**组织的，需要把两者对接起来：

```
1. 遍历输出（输入）的全部行：输出连续，行起始偏移恒为 row * xAxisSize
   （不需要二分，也不需要行起始偏移的单调性假设）
2. 逐行取与本核段相交的部分 [s, e) = [max(row*xAxisSize, gStart), min(row*xAxisSize+xAxisSize, gEnd))
3. 该输出行 → index 行号：
     index 前部各维与输入完全一致（fullCover）→ 就是 row 本身
     否则把 row 按**输入前部维**分解坐标（从最后一个前导维往前），任一坐标 ≥ index 对应维
       即该输出行没有 index 行（返回 -1）
4. 行内按 pieceCap 切片，每片：
     从输入搬入基准片（整片 RMW）
     if 该行有 index 行：
        for 每个 index 分片：
          搬入 index/src 片 → 向量化负索引修正 + 片基址重定位 → 标量 UB 散写
     整片回写输出
```

> **红线 3**：核内遍历的必须是**输出（输入）的全部行**，不能只遍历 index 的行。
> 当 index 前部维小于输入对应维时（如 `x=(8,32,16)` / `index=(4,16,8)` / `dim=1`），
> index 只覆盖输出前部维的一个**子网格**，只遍历 index 行会让其余输出行完全不被写入
> （实测 Max diff 5.24）。而本 Pattern 的输出**不靠调用方预填**，所以漏写的行会直接留脏值。
> 遍历全部输出行后，无 index 行的行也把输入原值整片回写，输出空间才被真正全覆盖。

- **输出行 → index 行的坐标分解必须从最后一个前导维往前**（`for d = preDimNum-1 .. 0`）。
  row 线性值中最后一个前导维变化最快（低位），先对最慢维取模会导致 3D+ 行映射错位
  （2D 只有一个前导维不受影响，容易漏测）
- `fullCover` 快路径省掉每行的坐标分解；不满足时每行只多一次「前部维数次取模/除法」，
  相对该行的 index 扫描可忽略

**标量散写实现**（`DoScatter<REDUCE>`，与 ops-nn `LoadCache8` / `DoScatterElementsOneByOne` 一致）：

```cpp
// UB 内裸指针访问：__ubuf__ 在设备构建展开为 __attribute__((cce_unif_buff))，CPU stub 构建展开为空
__ubuf__ int32_t* idxp = reinterpret_cast<__ubuf__ int32_t*>(idxLocal.GetPhyAddr());
__ubuf__ T_DATA*  srcp = reinterpret_cast<__ubuf__ T_DATA*>(srcLocal.GetPhyAddr());
__ubuf__ T_DATA*  yp   = reinterpret_cast<__ubuf__ T_DATA*>(yLocal.GetPhyAddr());
// 8 路展开：先批量取值再批量散写
#pragma unroll
for (k = 0; k < 8; k++) { rels[k] = (uint32_t)idxp[j+k]; vals[k] = srcp[j+k]; }
#pragma unroll
for (k = 0; k < 8; k++) { if (rels[k] < valid) yp[rels[k]] = vals[k]; /* reduce=1 时 += */ }
```

- 用 `LocalTensor::GetPhyAddr()` + `reinterpret_cast<__ubuf__ T*>` 而非 `GetValue(j)`：
  批量取值/写值，配合 8 路展开让标量流水吃满
- **8 路展开的作用**：把「读 index / 读 src / 比较 / 写 y」四类操作的依赖链打散，
  隐藏 UB 访问延迟

### 5. 负索引修正 + 片基址重定位（向量化批量，关键技巧）

标量散写需要「修正后的下标直接就是 y 片内下标」，所以把两件事一次向量化做完：

```
ShiftRight(scratch, idx, 31)         // 算术右移 31 位：负数 -> -1，非负 -> 0
Muls(scratch, scratch, -xAxisSize)   // 负数 -> -xAxisSize
Add(idx, idx, scratch)               // idx += (idx < 0) * xAxisSize   （负索引修正）
if (startInRow != 0)
    Adds(idx, idx, -startInRow)      // idx -= 片在该行内的起始偏移        （片基址重定位）
```

- **`startInRow` 是片在该行内的起始偏移，不是绝对偏移 `ps`**。这一步做完后
  合法下标恰好落在 `[0, pieceLen)`，标量侧可直接 `yp[rel]`
- **单次无符号比较同时完成上下界过滤**：`if ((uint32_t)rel < pieceLen)`。
  修正后为负的值（未落到本片）转成巨大无符号数，同样被挡掉，无需两次比较、无需钳位
- 向量长度按 8 元素对齐（int32 向量指令按 32B 块执行），尾部补齐元素在 UB 内不参与散写
- 越界 index 行为未定义（与 Gather 语义一致）
- **覆盖写的 last-wins 语义天然保持**：同一行内按「片升序 → index 分片升序 → 片内 j 升序」
  处理，与原逐元素实现的 j 升序一致

### 6. UB 预算与分片

```
SCATTER_UB_BUDGET = 176KB（910B 单核 UB 192KB，留系统余量）

idxChunk  = min(indexAxisSize, 8192, 65535/4)，再 8 对齐
    # 8192 兼顾同步开销与 UB 预算；65535/4 = DataCopyPad blockLen uint16 上限
pieceCap  = min(65535/sizeof(T), (BUDGET - idxAlloc*(8+sizeof(T)) - 256) / sizeof(T))
            再按 outBlockSize 下对齐；装不下时折半 idxChunk 重算
idxAlloc  = ceil_align(idxChunk, 8)
```

| Buffer | 大小 | 说明 |
|--------|------|------|
| yBuf | `pieceCap * sizeof(T)` | 基准片：从**输入**整片读入 / UB 内散写 / 整片回写输出 |
| idxBuf | `idxAlloc * sizeof(int32)` | index 片，修正后即为片内下标 |
| scratchBuf | `idxAlloc * sizeof(int32)` | 负索引修正的向量 scratch |
| srcBuf | `idxAlloc * sizeof(T)` | src 片 |

实测分配：fp32 → `idxChunk=8192, pieceCap=16368`（UB 用量 163904B）；
fp16 → `idxChunk=8192, pieceCap=32736`（UB 用量 147520B），均在 176KB 内。

- 单缓冲 + 显式事件同步（`TBuf<TPosition::VECCALC>` + `Get<T>()` + `SetFlag`/`WaitFlag`），
  **不要给 index/src 加双缓冲**——会挤占 `idxChunk` 使 index 分片轮数增加，净亏
  （与 gather mode1 的结论一致：瓶颈是同步轮数与搬运延迟，不是带宽）

**同步链（红线）**：

| 事件 | 位置 | 作用 |
|------|------|------|
| `MTE2_S` | 基准片（输入）搬入 → 标量 | 搬入完成才能散写 |
| `S_MTE2` | **每片 index/src 搬入前** | 上一片标量读未结束就覆盖 idx/src 会**稳定错位**（实测大 indexAxis 用例 4 个失败、位置固定在 27/33、输出 0.0） |
| `MTE2_V` | index/src 搬入 → 向量修正 | |
| `V_S` | 向量修正 → 标量散写 | |
| `S_MTE3` | 标量散写完 → 整片回写 | |
| `MTE3_MTE2` | 整片回写 → 下一片搬入 | 回写读完基准片缓冲才能重新覆盖 |

> **`DataCopyExtParams::blockLen` 是 `uint16_t`**（不是 uint32_t，CANN 9.0.0 实测）。
> 单次搬运 ≤ 65535 字节，超了会报 `-Wc++11-narrowing` 编译错，
> 所以片长必须由 host 侧按 `SCATTER_MAX_COPY_BYTES = 65535` 约束后再 `static_cast<uint16_t>`。

### 7. Host 侧红线：不要在 Host 做 index 范围校验（会触发 NPU→CPU 同步）

```cpp
// ❌ 禁止：触发 NPU→CPU 同步，Host 耗时从 ~20us 飙升到 9000+us
auto indexCpu = idxT.to(at::DeviceType::CPU).to(at::kLong);
for (int64_t i = 0; i < indexCpu.numel(); i++) {
    TORCH_CHECK(indexCpu[i] >= 0 && indexCpu[i] < xAxisSize, ...);
}

// ✅ 正确：不做数据级校验，或仅做 shape 级校验（无数据搬运）
//     index 范围校验应在 kernel 端完成（越界行为未定义，与 Gather 语义一致）
```

实测影响（`[1000, 512]` float32 scatter）：

| 方案 | 自定义耗时 | 标杆耗时 | 加速比 |
|------|-----------|---------|--------|
| 有 index 校验 | 1315us | 613us | 0.47 |
| **无 index 校验** | **348us** | 613us | **1.76** |

**对标杆底层算子（供对照）**：PyTorch 内置 `scatter_` 在 NPU 上调 `aclnnInplaceScatter`，
`scatter_add_` 调 `aclnnScatterAdd`：

| 场景 | 底层算子 | Device 耗时 | Host 耗时 |
|------|---------|------------|----------|
| [100,256] scatter | aclnnInplaceScatter | 32us | 15us |
| [1000,512] scatter | aclnnInplaceScatter | 562us | 18us |
| [100,256] scatter_add | aclnnScatterAdd | 105us | 17us |
| [1000,512] scatter_add | aclnnScatterAdd | 1974us | 20us |

## Tiling 字段

| 字段 | 含义 |
|------|------|
| reduce | 0 覆盖写 / 1 累加 |
| xTotal | 输出（= transpose 后的输入）元素总数，**分核总量必须用它** |
| preDimNum / xAxisSize / indexAxisSize | 前部维数 / 输入尾轴长 / index 尾轴长 |
| rows | 输出行数 = xTotal / xAxisSize（**遍历输出全部行用它，不要用 index 行数**） |
| blockNum | 实际使用核数 = min(ceil(xTotal/outBlockSize), coreNum) |
| outBlockSize | 输出分块粒度（元素）= 64B / sizeof(T)：fp32=16、fp16=32 |
| pieceCap | 单个 y 片元素数上限（UB 预算约束，≤ 65535/sizeof(T)） |
| idxChunk | index/src 单次搬入元素数（≤ indexAxisSize，≤ 8192，8 对齐） |
| xShape[] / indexShape[] | transpose 后输入 shape / index shape（输出行 ↔ index 行映射用） |

## 已验证场景（910B / CANN 9.0.0）

- 直调通路 **15/15 全通过**（max diff 全 0）：2D/3D × 尾轴/非尾轴（dim 0/1/-1）×
  FP32/FP16 × 覆盖写/累加 × 负索引 × 大轴专项（`2×30000`、`2×131072`）
- PyTorch 通路 **40/40 全通过**（max diff 全 0）：含 3D 非尾轴、smaller-index
  （index 前部维小于输入，同时验证 `xTotal` 分核与「输出全部行遍历」两处修正）、
  int64 index、`2048×2048`、`x8192 idx100000`、`x131072 idx131072`
- 语义对齐：torch CPU `torch.scatter` 对重复索引为 **last-wins**
  （实测 `x=zeros(3), idx=[1,1,1,0], src=[10,20,30,5]` → `[5,30,0]`），与本 kernel 一致；
  torch_npu 的重复索引结果是竞态，不能当 golden

**性能（优化后 vs 原生 torch，min-of-5 测量；2026-09-28 复测，已去掉输出 clone）**：

| 用例 | custom | torch | 对比 | 优化前 |
|------|--------|-------|------|--------|
| scatter 1024×4096 | 0.49ms | 1.60ms | **快 3.28x** | 0.45x |
| scatter 4096×4096 | 1.88ms | 6.46ms | **快 3.43x** | 0.47x |
| scatter 2×8192 idx100000 | 0.55ms | 1.63ms | **快 2.98x** | 0.41x |
| scatter 2×131072 idx300000 | 1.63ms | 4.83ms | **快 2.96x** | 0.40x |
| scatter_add 1024×4096 | 1.58ms | 3.08ms | **快 1.95x** | 0.70x |
| scatter_add 4096×4096 | 6.25ms | 12.32ms | **快 1.97x** | 0.71x |
| scatter_add 2×8192 idx100000 | 0.60ms | 1.65ms | **快 2.76x** | 0.41x |
| scatter_add 2×131072 idx300000 | 1.77ms | 4.89ms | **快 2.76x** | 0.39x |

**优化路径（按收益排序）**：

1. **UB 内 RMW 替代 GM 标量写（决定性，0.4x → 3x）**：把散射目标从 GM 换到 UB，
   代价是与输出等量的额外读 + 回写搬运，收益是标量写延迟降一个数量级
2. **8 路展开 UB 散写**：打散依赖链，隐藏 UB 访问延迟
3. **向量化「负索引修正 + 片基址重定位」**：把每元素的两次标量判断挪到向量流水
4. **fullCover 快路径**：index 前部各维与输入一致时输出行号即 index 行号，免每行坐标分解
5. **输出不做预填（去掉 `clone`）**：kernel 读**输入**片做 RMW 基准，省一次与输出等量的
   设备拷贝。实测 `.clone()` 代价（min-of-5）：`1024×4096` 约 18us、`4096×4096` 约 31us
   （输出 67MB 命中 192MB L2）、`2×131072` 约 20us，占比 1%~4%。收益不大，但消除了
   「输出必须由调用方预填」的耦合，以及 clone 进 torch_npu 延迟队列后对直调 kernel 的可见性风险

**残余代价（已知）**：index 行会被其覆盖到的**每个**核完整扫描一遍。核数远多于行数时
（如 `2×131072` 用 40 核，每核只负责 1/20 行），有约 20x 的 index 冗余扫描
（40 核 × 300000 = 12M 次迭代）。这是「按输出空间分核」换取 cache line 安全与
累加确定性的必然代价；但写已从 GM 变 UB，仍有 2.9x 收益。

> **测量纪律**：NPU 设备波动可达 ±30%，必须多轮取最小值
> （`bench(fn, iters=50, warmup=20, rounds=5)`），单次平均值会误判优化方向。

## 与相邻 Pattern 的区分

- P1（扁平索引取数）/ P2（沿轴索引取数）：index 决定**从哪里读**，输出连续、可向量化搬出；
  Scatter 是 index 决定**写到哪里**，输出地址分散，只能标量写——但**标量写的目标应在 UB**，
  这是本 Pattern 与 Gather 系性能差距能否抹平的关键
- P3 / P4：P3 覆盖写（无冲突或 last-wins），P4 累加（需归约）。
  本实现两者共用同一分核与核内流程，仅散写语句不同（`=` vs `+=`），
  因为「输出元素单属主」分核已同时解决冲突问题
