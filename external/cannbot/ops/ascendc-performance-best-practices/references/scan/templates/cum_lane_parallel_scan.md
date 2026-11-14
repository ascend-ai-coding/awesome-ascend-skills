# Lane 并行顺序扫描（消掉 log2(R) 放大）

对应模板：[dav310/cum_lane_parallel_scan.template](dav310/cum_lane_parallel_scan.template)

## 什么时候用

已经用[行内 Kogge-Stone](cum_row_kogge_stone.md) 把搬运粒度摆平之后，
**扫描本身的向量指令条数成了瓶颈**时用这个替掉 Kogge-Stone。

判断方法：把 `indices` 的搬出（int64，占总流量 2/3）临时注掉跑一次。
**耗时不变就说明纯算力受限**，此时本模板的收益最大。cann-bench Cummin 上
fp32/int32 用例（每 vreg 只有 64 lane，log2 放大最狠）实测提速 25%~61%。

前置条件：

- 扫描段已经完整躺在 UB 里（本模板不管搬运，只换 UB 内的算法）
- **比较类型必须是 4 字节**：fp16/bf16 一律升 fp32 再进来（见下节），索引随之恒为 `int32_t`。
  索引与比较类型必须等宽，不等宽时索引侧要拆包成两个寄存器，多出来的开销会把收益吃掉。
- `R > 2 * VLC`。再小的话相位二那固定的 `log2(VLC)` 级反而不划算，留在 Kogge-Stone。

### ⚠️ 16 位输入必须升 fp32（这条推翻了 Kogge-Stone 时代的结论）

「fp16/bf16 用原生 16 位比较」在 Kogge-Stone 路径上是赚的（每 vreg 128 lane、
进出各省一趟 Cast，实测快 15%~21%），**换到本模板后完全反转**。
同为 `1152×1024` 个元素的实测：

| | Kogge-Stone | lane 并行 |
|---|---|---|
| fp16（原生 16 位比较） | 18.55µs | 15.05µs |
| fp32 | 28.58µs | **9.27µs** |

Kogge-Stone 下 fp16 比 fp32 快 1.54 倍（符合 128 vs 64 lane 的预期），
lane 并行下却慢 1.62 倍。注意 fp16 的 gather 指令数只有 fp32 的**一半**（36 vs 68）、
元素触达数也相当，却慢近两倍——**方向反了就只能是 gather 本身的位宽路径问题**：
16 位的 `Gather`/`Scatter` 没有全速路径。
排查时先用 `ResourceConflictRatio` 排除了 bank 冲突（`bank_cflt` 仅 0.1%）。

改成一律升 fp32 后 fp16 降到 9.39µs、与 fp32 打平，6 个 16 位用例快 25%~38%（+1.68 分）。
代价只是进出各一趟 Cast（`LaneFlatten` 顺带做展宽，出口用 `LaneCastOut` 收窄），
远小于 gather 的损失。

**教训：换了扫描算法，比较类型的选型必须重测，旧结论不会自动跟着算法迁移。**
注意对照 [cum_tile_resident_scan.md](cum_tile_resident_scan.md)——那条路访存连续、不走 gather，
16 位反而可以边搬边 Cast，结论相反。

## 为什么比 Kogge-Stone 快

| | 向量指令条数 | 深度 |
|---|---|---|
| Kogge-Stone | `log2(R) · (R/VLC) · ~14` | `log2(R)` 级 |
| 本模板 | `~15 · (R/VLC) + 14 · log2(VLC)` | 固定 |

`R/VLC` 那一项两者都有，差别在 Kogge-Stone 多乘了 `log2(R)`。
`R = 2049`、fp32（VLC=64）时是 12 倍 vs 常数，这就是 61% 的来处。

## 三个相位

把 `VLC` 条 lane 各分一段**连续**区间，段长 `S = ceil(R / VLC)`：

```
lane 0: [0, S)      lane 1: [S, 2S)     ...     lane VLC-1: [(VLC-1)S, VLC·S)
```

```
相位一  每条 lane 沿自己那段顺序推进 S 步，得到 VLC 个段结果
相位二  对这 VLC 个段结果做扫描，拿到每条 lane 的 exclusive 前缀
相位三  每条 lane 带上前缀重扫自己那段 S 步，就地写回值与索引
```

段内元素在 UB 上按 lane 跨步（相邻 lane 差 `S` 个元素），所以相位一/三的存取
只能用 `Gather` / `Scatter`——**这就是所谓「虚拟转置」**：
不真的搬数据，用索引寄存器让一条 vgather 指令横跨 VLC 个段各取一个元素。

lane l 覆盖的是位置上靠后的连续区间，**按 lane 序合并即按位置序合并**，
所以相位二能直接沿用同一个 `combine`，不必为不满足交换律的 combine
（并列取更晚、NaN 粘滞）额外裁决。

## 关键实现细节

### 1. 段长必须避开 bank 冲突

段长 `S` 同时就是 lane 间的 UB 地址步长。**步长取 2 的幂时 VLC 条 lane 的
gather 全撞在同一批 bank 上**，实测 `R=4096` bf16（S=32）比错位版本慢 5.5 倍
（160 µs → 887 µs）。

规则：`S * sizeof(C)` 是 16 的整数倍时取奇数错开（`S |= 1`）；
否则保持原样。步长本来就细碎的情况（如 `S=4`、fp16 → 8 B）不必错位，
因为错位会把补齐区从 `R` 撑到 `VLC·S`，白算一截——`R=511` 时错位反而慢 15%。

```cpp
int64_t SegLen(int64_t r, int64_t lanes, int64_t cBytes)
{
    int64_t s = CeilDiv(r, lanes);
    return (s * cBytes % 16 == 0) ? (s | 1) : s;
}
```

### 2. 补齐区要预填幺元

段长统一成 `S` 之后数据区是 `VLC·S ≥ R`，尾部那截必须填幺元，
这样相位一/三的每一步都是满掩码，不用逐步算尾部掩码。
做法是先整块铺幺元再用掩码覆盖真实数据（见模板里的 `GsFlatten`）。

正确性上不必担心补齐区污染结果：位置 `< R` 的元素都落在 lane `≤ lLast`，
而 lane `< lLast` 是满的；lane `lLast` 内部顺序推进到 `R-1` 时还没碰到补齐区。

### 3. 相位二留在寄存器里做

`VLC` 个段结果正好一个 vreg。用 UB 暂存 + 复用 `KsLevel` 做 `log2(VLC)` 级
虽然能跑，但每级都要往 UB 回写再读，实测 `R=511` 的用例因此**倒退 19%**。

正解：**三个相位塞进同一个 `__VEC_SCOPE__`**，段结果全程留在寄存器，
lane 间平移用寄存器间的 `Gather`（vselr）：

```cpp
Arange(shIdx, (I)(-d));                          // lane l → l-d
Gather(vA, curMin, (RegTensor<G>&)shIdx);         // 寄存器内按 lane 平移
Compare<I, CMPMODE::LT>(before, laneReg, dReg);   // lane < d 的取值无效
Or(takeB, takeB, before);                        // 无效时判成「保持自己」
```

lane `< d` 平移后取到越界 lane，内容是垃圾，用 `before` 掩码判成「保持自己」
即可——这是寄存器内的置换，不访存，不会触发地址异常。

最后把 inclusive 前缀右移一格得到 exclusive，lane 0 取幺元（或块间进位）。

### 4. 两行交错填指令延迟

每一步都依赖上一步的 `(curMin, curIdx)`，单行的依赖链把指令延迟全暴露出来，
实测约 **4.6 cycle/指令**。把两行塞进同一个 vec scope，两条独立的依赖链互相填空，
实测再快 8%~18%（`R` 越小、行数越多收益越大）。

寄存器预算：每行独占 `cur/x/vA/curIdx/gIdx/iA` 共 6 个，
`laneReg/shIdx/dReg/identReg/zeroI` 共享。两行合计 17 个 vreg + 8 个掩码，够用。

模板中已提供完整实现 `LaneParallelScanPair`（见 `cum_lane_parallel_scan.template`），
三个相位与单行版完全对应，变量各开两套（0/1 后缀）。
在行批处理 `BatchCompute` 中按两行一组调用，奇数尾行退回单行 `LaneParallelScan`。

### 5. 块间进位

段扫描不是整行时（跨核借 R、或 `R > rChunk` 分块）要接住上一块的进位。
**先把进位并进 lane 0 的段结果**，相位二的 lane 间扫描会自动把它一路带到所有 lane；
再在 exclusive 右移那一步把 lane 0 单独置成进位本身。

不要在相位二之后对所有 lane 统一并一次进位：lane 0 的 exclusive 是幺元，
`combine(carry, identity)` 在 carry 也是幺元时会取到下标 0 而不是 carry 的下标。

### 6. `LocalMemBar` 必须写在 vec scope 内

相位三 scatter 写回的前缀随后要被 MTE3 搬走，需要一道
`LocalMemBar<VEC_STORE, VEC_LOAD>()`。**写在 `__VEC_SCOPE__` 外面能编译通过，
运行时报 `device error type 3, error code 507035`**，而且只在数据量过阈值后才崩，
极易误判成 gather/scatter 的地址范围限制。

## 会踩到的编译期限制

| 限制 | 绕法 |
|---|---|
| `Arange` 不支持无符号类型 | gather 索引要 u16/u32，用同宽有符号寄存器算好再在调用点强转 |
| 寄存器间没有 `DataCopy` | 复制一个 `RegTensor` 只能重算（`Muls`/`Arange`） |
| `int16 → int64` 不能一步 `Cast` | 编译通过但下标错，必须逐级 2 倍展宽。**升 fp32 后索引恒为 int32，这个坑连带消失** |
| gather 索引宽度须与数据宽度一致 | 由 `vgather2` 的类型约束决定。比较类型恒为 4 字节，故索引恒配 `uint32_t` |
| vec scope 内不要写数据相关分支 | `LaneFlatten` 只给尾块铺幺元本该更快，写成主循环里的 `if (j+1==dataLoops)` 却慢 **40%**——分支打断软流水。把尾块处理提成独立小循环即恢复 |
