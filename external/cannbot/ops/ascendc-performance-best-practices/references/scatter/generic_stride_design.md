# 实现⑥：通用路径（三张量非 dim 维范围不等）

> 总览与选型见 [scatter_design.md](scatter_design.md)。
> 一句话：**torch 只要求 `index.size(k) ≤ self.size(k)`，非 dim 维允许 indices/updates 比 data 小。
> 这时沿 dim 的步长不再是三者通用的 `inner`，必须按每维独立 stride 还原坐标。**

## 1. 为什么必须有这条路径（先读这段）

①~⑤ 五条快路都建立在一个隐含假设上：**三张量非 dim 维范围全等**，因而

- `inner` 是三者通用的、沿 dim 的步长都是 `inner`
- `updates` 与 `indices` 共享同一个线性下标 `i`

torch 的实际语义比这宽：`data` 可以是 `[2,2]`、`indices` 和 `updates` 是 `[1,2]`。
**这类形状在很多公开用例集里恰好一条都没有**，凭手上的用例是测不出来的。

> 真实代价：某次提交据公开的 20 条用例判断"非 dim 维必然相等"，直接 `TORCH_CHECK` 拒了这类输入，
> 结果隐藏用例 **77/80、correctness 门禁整体判失败、掉 15 分**。
>
> **算子的合法输入域要按语义规范推（`desc.md` / `proto.yaml` / torch 文档），不能按手上的用例推。**

Host 侧判据：

```cpp
bool extentsMatch = true;
for (int64_t i = 0; i < rank; ++i) {
    TORCH_CHECK(idxC.size(i) <= dataC.size(i) && idxC.size(i) <= updC.size(i), ...);
    if (idxC.size(i) != updC.size(i) || (i != d && idxC.size(i) != dataC.size(i))) extentsMatch = false;
}
const bool useGeneric = !extentsMatch && needScatter;
```

## 2. 寻址模型

```
dataOff = Σ_{k≠dim} coord_k · dataStride[k]        // 非 dim 部分
dst     = dataOff + ix · dataStride[dim]
updOff  = Σ_k coord_k · updStride[k]               // updates 有自己的 stride
```

坐标 `coord_k` 从 `indices` 的线性下标 `i` 按 `idxShape` 逐维取模还原：

```cpp
__simt_callee__ __aicore__ inline void GenDecompose(const ScatterGenMeta& m, int64_t i,
                                                    int64_t& dataOffNoDim, int64_t& updOff, int64_t& j)
{
    int64_t rem = i;
    dataOffNoDim = 0; updOff = 0; j = 0;
    for (int32_t d = m.rank - 1; d >= 0; --d) {
        const int64_t ext = m.idxShape[d];
        const int64_t c = rem % ext;
        rem /= ext;
        updOff += c * m.updStride[d];
        if (d == m.dim) j = c;
        else            dataOffNoDim += c * m.dataStride[d];
    }
}
```

## 3. meta 的传递

rank / dim / 三组 stride 是运行期才知道的变长数据，用一个**定长 int64 设备缓冲**传进 kernel，
不要塞进 kernel 形参（rank ≤ 8 时会是 26 个标量参数）：

```
meta 布局（int64 × 26）：[0]=rank  [1]=dim  [2..9]=idxShape  [10..17]=dataStride  [18..25]=updStride
```

Host 侧一次性搬上去：

```cpp
std::vector<int64_t> meta(SCATTER_GEN_META_LEN, 0);
meta[0] = rank; meta[1] = d;
for (int64_t i = 0; i < rank; ++i) {
    meta[2 + i]      = idxC.size(i);
    meta[2 + 8 + i]  = dataC.stride(i);
    meta[2 + 16 + i] = updC.stride(i);
}
metaT = torch::from_blob(meta.data(), {SCATTER_GEN_META_LEN}, torch::kInt64).clone().to(dataC.device());
```

Kernel 侧每个线程开头 `GenLoadMeta` 一次，读进寄存器/栈上的结构体，循环内不再碰 GM。

## 4. 仲裁策略与快路一致

| reduce | 走法 |
|---|---|
| `none` | **两趟胜者**（同[实现③](element_parallel_design.md#3-两趟胜者)）：`atomic_max(win + dst, j+1)` → `win[dst] == j+1` 才写 |
| `add` / `amin` / `amax` / `multiply` | 数值与确定性契约允许时可用直接原子（同[实现④](element_parallel_design.md#4-直接原子需满足数值契约的-reduce)），multiply 仍用 CAS 循环 |

注意与快路③的一处差别：这里记的是 **`j+1`** 而不是 `i+1`。因为 `updates` 有自己的 stride，
`i` 不能直接当 `updates` 的下标用，趟 2 无论如何要重算 `updOff`，记 `i` 就没有省事的价值了。
而对固定 `dst`，`j` 同样是严格递增且唯一的（核心不变量），足以裁决。

## 5. 这条路径**没有**做的优化（当前状态与下手点）

**设计取向是正确性优先**，性能上刻意保守：

| 保守点 | 现状 | 可能的收益 |
|---|---|---|
| 索引运算宽度 | 一律 int64，不做 int32 收窄 | 低（快路实测收窄零收益） |
| 坐标分解 | 每元素逐维 `%` + `/`，rank 次循环 | 中。可按 rank 编译期特化（`if constexpr (RANK == k)` 展开），消掉运行期循环与可变维度的寄存器压力 |
| meta 访问 | 每线程从 GM 读一次 26×8B | 低（一次性） |
| 线程数 | 固定 1024 | 未调 |

> 隐藏/生产用例里走这条路的比例可能远高于公开用例（公开集常常一条都没有），
> **它很可能是隐藏集分数的大头嫌疑**。本地无法复现隐藏用例，但可以照 §1 的形状自己造性能测试。

## 6. 与快路的互斥（必须显式保证）

> ⚠ **②③④⑤ 的判据都必须带 `extentsMatch`。** 这是踩过的真实缺陷：
> `usePackArb` 漏了它，于是 "2 字节 + none + starved" 时快路与通用路径**同时成立**，
> `y` 被重绑到 workspace 视图，而通用分支在派发函数里排在最前、按连续下标写 —— 输出以 2 倍元素步长错位。
> 公开的 20 例不触发通用路径，**标准集满分也照样挂隐藏集**。
>
> **加任何新快路，第一件事是确认它排除了 `!extentsMatch`。** 回归见 [precision_and_traps.md §5](precision_and_traps.md#5-回归网怎么搭)。
