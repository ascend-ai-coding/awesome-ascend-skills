# 标量降级

## 概述

标量降级是指：编译器把本该一次处理多个数据的向量操作，降级成逐个元素执行的标量循环。性能通常暴跌 10~100 倍。

## 适用条件

1. 通用算术操作（VAddOp, VSubOp, VMulOp, VMinOp, VMaxOp, VAbsOp, VShLOp, VShROp, VInterleaveOp, VDeinterleaveOp）

```
┌────────────────────────┬──────────┐
│    数据类型             │ 是否降级 │
├────────────────────────┼──────────┤
│ f16, f32, i8, i16, i32 │ 不降级   │
├────────────────────────┼──────────┤
│ i64                    │ 降级     │ 
└────────────────────────┴──────────┘
```
总结：输入类型为i64时，这些算术操作都会降级为标量循环。


---
2. 比较操作（大于、小于、等于等）
```
┌──────────────┬────────────────────────┬───────────────────────────────┐
│   数据类型    │ EQ / NE（等于/不等于）  │ LT / GT / LE / GE（大小比较）  │
├──────────────┼────────────────────────┼───────────────────────────────┤
│ f16, f32     │ 不降级                 │ 不降级                        │
├──────────────┼────────────────────────┼───────────────────────────────┤
│ i32          │ 不降级                 │ 降级                          │
├──────────────┼────────────────────────┼───────────────────────────────┤
│ i8, i16, i64 │ 降级                   │ 降级                          │
└──────────────┴────────────────────────┴───────────────────────────────┘
```
总结：只有 i32 的等于/不等于 和 所有浮点比较 能走向量加速；其余整数比较都会降级。

---

---
3. 取余操作

始终降级。建议将`a % b` 替换成i32类型的 `a - (a // b) * b`

---

4. 扩展乘法（vmulext））

始终降级。该操作在 IR 层面只支持 i32，而 i32 触发降级，所以实际上没有向量硬件支持。

---
5. 累积操作（cumsum / cumprod）
```
┌───────────────────────────────────┬────────────────────┬──────────────────────┬──────────────┐
│             数据类型              │ 累积维度是最后维度 │ 累积维度不是最后维度 │ 多个累积维度 │
├───────────────────────────────────┼────────────────────┼──────────────────────┼──────────────┤
│ f16 / f32 / bf16 / i8 / i16 / i32 │ 降级               │ 不降级               │ 不降级       │
├───────────────────────────────────┼────────────────────┼──────────────────────┼──────────────┤
│ i64                               │ 降级               │ 降级                 │ 不降级       │
└───────────────────────────────────┴────────────────────┴──────────────────────┴──────────────┘
```
总结：
- i64 类型的累积操作无条件降级。
- 非 i64 类型下，如果累积维度是张量的最后一个维度，就会降级。
- 多个累积维度不会触发降级（但可能有其他限制）。

---
6. 归约操作（reduce）

归约的降级条件取决于芯片架构：

Mem-based 架构（如 Ascend910B 系列）：
- sum / prod / max / min / xori：仅 i64 时降级。
- argmax / argmin（带索引的极值归约）：
- 整数类型（i16 / i32 / i64）：降级
- 浮点类型（f16 / f32 / bf16）且 flatten 后维度 > 2：降级
- 浮点类型且 flatten 后维度 ≤ 2：不降级
- any / all / ori / andi / none：不降级

Reg-based 架构（如 Ascend310B / Ascend950）：
- 基本归约（sum / prod / max / min 等）：不降级
- argmax / argmin：仅在内存访问不对齐时降级

## 优化建议

1. 根据上述数据类型的降级条件，调整数据类型，避免命中降级条件；
2. 避免使用扩展乘法相关算子；
3. reduce算子的输入尽可能32B对齐；
4. 避免使用取余算子，建议将`a % b` 替换成i32类型的 `a - (a // b) * b`；

### cumsum和cumprod的累积维度降级优化方法

一维张量 `cumsum(axis=0)`时必定会触发降级 ，因为唯一的维度就是最后一个维度。

**解决办法**：把一维数据**折叠成二维 (SUB_N, COLS)**，让 `cumsum` 在 `axis=0` 上跑。二维张量 flatten 后有两个维度，`axis=0` 不再是最后一个维度，就能走硬件向量加速。

#### 适用条件

- 代码中对一维 slice 调用了 `tl.cumsum(x, axis=0)` 或 `tl.cumprod(x, axis=0)`。
- 非 `i64` 类型（`i64` 无论怎么 reshape 都会降级）。

#### 核心思想

```python
# ❌ 一维张量：唯一的维度就是最后维度，触发标量降级
x_cumsum = tl.cumsum(x_1d, axis=0)

# ✅ 折叠成二维 (SUB_N, COLS)，cumsum 在 axis=0，避开最后维度
# 例子：SUB_N=4, COLS=3，一维下标 [0,1,2,3,4,5,6,7,8,9,10,11] 重排为列主序
#       col0   col1   col2
# row0    0      4      8
# row1    1      5      9
# row2    2      6     10
# row3    3      7     11
# 这样 axis=0 是"从上到下累加"，不是最后维度

# ⚠️ 必须先从 GM 连续加载为一维，再在寄存器里 reshape/trans，不要构造 col_major_idx 做跨步加载
x = tl.load(x_ptr + tl.arange(0, BLOCK_N))       # 1D 连续加载
x_col = tl.trans(tl.reshape(x, (COLS, SUB_N)))   # 寄存器内重排为列主序，shape (SUB_N, COLS)
cumsum_col = tl.cumsum(x_col, axis=0)            # ✅ axis=0 不是最后维度，走向量加速

# 列间补偿：第 j 列的结果要加上前面所有列的总和
# 例如 col1 的每个元素都要加上 col0 的总和
col_sums = tl.sum(x_col, axis=0)                 # 每列一个总和，shape (COLS,)
col_prefix = tl.cumsum(col_sums, axis=0) - col_sums  # [0, sum(col0), sum(col0)+sum(col1), ...]
y = cumsum_col + col_prefix                       # 块内 cumsum + 列间偏移 = 全局正确结果

# 转置回行主序，再 flatten，保证写回内存连续
y = tl.reshape(tl.trans(y), (BLOCK_N,))
```

#### 完整示例（含跨块累加）

```python
# ⚠️ 不要在 GM 层面构造 col_major_idx 做跨步加载，会大幅降低访存带宽并撑爆 UB
# 正确做法：先 1D 连续加载，再在寄存器内 reshape/trans

for n_start in range(0, N, BLOCK_N):
    offsets = n_start + tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # 1D 连续加载
    m = tl.load(mask_ptr + offsets, mask=mask, other=0)
    masked_x = tl.where(m, x, 0.0)

    x_col = tl.trans(tl.reshape(masked_x, (COLS, SUB_N)))  # 寄存器内重排为列主序

    cumsum_col = tl.cumsum(x_col, axis=0)                 # 块内前缀和
    col_sums = tl.sum(x_col, axis=0)
    col_prefix = tl.cumsum(col_sums, axis=0) - col_sums   # 列间偏移补偿

    y = cumsum_col + col_prefix                           # 块内全局正确
    y = tl.reshape(tl.trans(y), (BLOCK_N,))
    y = y + acc                                           # 加上前面所有块的总和

    tl.store(y_ptr + offsets, y, mask=mask)
    acc = acc + tl.sum(masked_x)                          # 更新跨块累积值
```

#### 关键点

1. **为什么要转二维**
   - 编译器判断降级只看 flatten 后的维度。一维时 `cumDim=0 == lastDim=0`；二维时 `cumDim=0 != lastDim=1`，条件不成立。

2. **col_prefix 为什么这样计算**
   - `tl.cumsum(col_sums)` 得到 `[col0, col0+col1, col0+col1+col2, ...]`。
   - 减去自身 `col_sums` 得到 `[0, col0, col0+col1, ...]`，这正是每列需要额外加的偏移量。

3. **为什么要 trans + reshape？**
   - `cumsum_col` 是列主序排列的，直接 flatten 会得到 `[0,1,2,3,4,5...]` 吗？不会。
   - 列主序 flatten 后是 `[0,4,8,1,5,9,2,6,10,3,7,11]`，和原始内存顺序不一致。
   - 先 `trans` 成行主序，再 `reshape`，才能得到正确的 `[0,1,2,3,4,5,6,7,8,9,10,11]` 顺序写回。

4. **参数选择**
   - `BLOCK_N = SUB_N * COLS`，两者建议取 64、128 等向量宽度的倍数。

5. **避免在 GM 上用索引张量做跨步加载**
   - 错误做法：`col_major_idx = tl.arange(...)[:, None] + ...; tl.load(ptr + col_major_idx)` 会在 GM 上产生跨步访存，且 `col_major_idx` 本身作为中间张量会占用大量 UB，极易触发 `ub overflow`。
   - 正确做法：先用 1D `tl.arange` 连续加载到寄存器，再用 `reshape` + `trans` 在寄存器内完成列主序重排，不产生额外 GM 访存开销。

6. **UB 溢出时不要盲目缩小 BLOCK_N**
   - 如果编译报 `ub overflow`，优先检查是否构造了过大的中间索引张量（如 `col_major_idx`、额外的 `valid_mask` 等），通过寄存器内 reshape 消除这些张量来释放 UB。
   - 盲目把 `BLOCK_N` 从 2048 缩小到 512 会导致循环次数翻 4 倍，loop overhead 完全抵消 cumsum 向量加速的收益，甚至严重劣化。
