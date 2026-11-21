# Scatter 族 — TilingData 字段语义

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

> 配合 [tiling-flow.md](tiling-flow.md) 使用：那里讲怎么算，这里讲算出来的每个字段是什么、谁消费。

> **形态说明**：Scatter 族的 kernel 是**多个独立 pass**（SIMD clone/cast + 若干 SIMT 散射 pass），
> 不是单一 kernel 吃一份 TilingData。direct-launch 形态下这些字段作为 kernel 实参逐 pass 传入；
> 注册算子工程里则合成一个 TilingData 结构 + tilingKey。字段语义两种形态一致。

---

## 1. 形状字段（合轴产物，全路径共用）

| 字段 | 类型 | 语义 | 来源 |
|---|---|---|---|
| `dim` / `d` | int32 | 归一后的 scatter 轴，`d = dim < 0 ? dim + rank : dim` | 属性 |
| `inner` | int64 | `prod(indices.shape[d+1:])`，一条纤维内沿 dim 的步长 | 合轴 |
| `outer` | int64 | `prod(indices.shape[0:d])` | 合轴 |
| `numFibers` | int64 | `outer × inner`，**纤维路径的并行宽度上限** | 合轴 |
| `sIdx` | int64 | `indices.shape[d]` | 合轴 |
| `sUpd` | int64 | `updates.shape[d]` | 合轴 |
| `sVar` | int64 | `data.shape[d]`，index 的合法上界 | 合轴 |
| `epTotal` | int64 | `outer × sIdx × inner` = `indices.numel()`，**元素级路径的并行宽度** | 合轴 |
| `numelData` | int64 | `data.numel()`，输出规模、workspace 尺寸基准 | 合轴 |

> **合轴只用 `indices` 的 shape**。用 `data.shape` 会在"非 dim 维 indices 比 data 小"时算出错误的 `inner`。

## 2. 路径与语义字段

| 字段 | 类型 | 语义 | 消费方 |
|---|---|---|---|
| `red` | int32 | reduce 模式：`0=none 1=add 2=mul 3=amin 4=amax`。**作为运行期标量在核内 switch 分发到 `if constexpr` 特化的 VF** —— 若提到编译期会让 `dtype × idx 宽度 × reduce × 线程数` 爆出几十份二进制 | 所有散射 pass |
| `esize` | int32 | 值字节宽（2/4/8），决定键宽与 clone 的实例化 | 选路 / clone |
| `extentsMatch` | bool | 三张量非 dim 维范围是否全等。**四条快路的判据必须全部带它**（详见 tiling-flow §2.2） | 选路 |
| `starved` | bool | `numFibers < coreNum × 1024` | 选路 |
| `narrowIdx` | int32 | 1 = 所有线性下标 `< 2^31`，索引算术可用 int32。**实测零收益**，保留供寄存器压力敏感的 VF 复用 | 所有散射 pass |
| `jBits` | int32 | `bitWidth(sIdx)`，⑤ 路径中 index 在 32 位键内的起始位 | ⑤ |
| `denseWinResolve` | bool | `epTotal × 4 ≥ numelData`。③ 的 apply pass 按 `dst` 直扫（省坐标分解）而非按更新元素扫。实现上以 `outer == 0` 作标记传入 | ③ |
| `useF32Acc` | bool | 半精度的累积型 reduce（`add`/`multiply`）需要 fp32 累加器 | ④/①/⑥ |

## 3. 启动配置字段（每个 pass 一组）

| 字段 | 类型 | 语义 |
|---|---|---|
| `blocks` | int64 | 该 pass 的 block 数，一律 `min(coreNum, ...)` |
| `threads` | int32 | 该 pass 每 block 的线程数。**只能取 `{1024, 512, 128, 32, 8}`** —— `LAUNCH_BOUND` / `dim3` 要求编译期常量，核内按 `switch (threads)` 展开 |

各 pass 的取值来源见 [tiling-flow.md §3](tiling-flow.md#3-各路径的-tiling-推导)。要点：

- ① 纤维独占：`threads` 由"能铺满核的最大候选值"决定，`blocks = min(coreNum, ceil(numFibers/threads))`
- ②③④⑤⑥：`threads` 按 pass 固定（`add` 与两条仲裁 pass 是 **256**，其余 1024），`blocks = min(coreNum, ceil(count/1024))`
- ⑤：`blocks = min(coreNum, numFibers)`（一条 warp 一条纤维，每 block 32 个 warp）

## 4. SIMD clone / cast pass 的字段

| 字段 | 类型 | 语义 |
|---|---|---|
| `copyElems` | int64 | 拷贝单元数。**int64 按 `numelData × 2` 走 4 字节版本** |
| `copyBlocks` | int64 | `min(coreNum, ceil(copyElems / 1024))` |
| `copyBlockLen` | int64 | `ceil(copyElems / copyBlocks)`，每核连续分片长度 |
| `copyTile` | uint32 | `floor(ubSize / 2 / 2 / 512) × 512`。**必须 512B 对齐**，否则 `DataCopyPad` 起址/长度不对齐 |
| `castBlocks` | int64 | 半精度 cast pass 的 block 数，tile 固定 16384 元素 |

## 5. workspace 字段

| 字段 | 尺寸 | 路径 | 说明 |
|---|---|---|---|
| `ws` | `4 × numelData` | ② | int32 打包键数组 |
| `win` | `4 × numelData` | ③ / ⑥(`none`) | int32 胜者数组，**必须预置 0**（表示"无人写"，保留 clone 好的原值） |
| `acc` | `4 × numelData` | 半精度累加 | fp32 累加器 |
| `meta` | 208 B（`int64[26]`） | ⑥ | `[0]=rank [1]=dim [2..9]=idxShape [10..17]=dataStride [18..25]=updStride` |

> ②③ 的 workspace 按 **`numelData`（输出规模）** 而不是 `epTotal`（更新规模）——键/胜者是按输出位置存的。

## 6. ⑥ 通用路径的 meta 布局

rank / dim / 三组 stride 是运行期变长数据，用**定长 int64 缓冲**传入，不要塞成 26 个 kernel 形参。
kernel 侧每线程开头读一次进栈上结构体，循环内不再碰 GM。

```
下标:  0      1     2..9        10..17         18..25
内容: rank   dim   idxShape[8]  dataStride[8]  updStride[8]
```

`rank < 8` 时高位补 0；`GenDecompose` 只遍历到 `rank-1`。
