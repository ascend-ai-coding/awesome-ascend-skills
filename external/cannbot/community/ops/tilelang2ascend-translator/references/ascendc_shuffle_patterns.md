# AscendC 重排/搬运类算子实现指南（Shuffle / Gather / Broadcast 通用）

> 适用范围：交织重排、奇偶拆分、gather/scatter、广播消费等**数据搬运主导**的
> 矢量算子（RoPE 交织、permute 类变体等）的 AscendC **生成与实现**。
> 全部条目来自实测，只给规则与量级——生成时按此写，一次到位。

## 0. 生成第一约束：固定开销

搬运/重排类算子的耗时大头常是**每-launch 固定开销**（建表、分发、启动），
不是搬运带宽：一个 64 元素的 case 实测 25µs，其中有效搬运不到 1µs。

因此生成结构时先算固定成本账：**任何"每次 launch 都要重建/重算"的结构
都要质疑**（offset 表、index 缓冲、多余的 kernel 拆分）。小 shape 下它直接
决定性能档位。

> **计时红线**：本文所有量级结论都来自 device 侧 kernel 时间（msprof
> Task_Duration）。若用 torch_npu Event 墙钟复核，读数会含 host 下发间隙与
> 共享设备干扰，小算子可能得出完全相反的结论（实测发生过 ~100x 虚高事故）。

## 1. device 侧结构选择（生成时直接照此写）

按实测收益量级排序：

### 1.1 规律重排优先用硬件 pattern 指令，不用"索引表 + Gather"

规律的元素重排（奇偶拆分、隔点提取、行内半区对调等）**先找硬件模式指令**
（如 stride-2 提取类的 GatherMask）：一次指令完成重排，零 offset 表。
反面结构：为规律 pattern 用 Gather + 每 launch 构建 offset 表——仅建表
实测 ~17.5µs/launch，比很多 case 的全部有效工作还贵。

旋转/半区对调（RoPE / RotaryMul 的 `half`，`rotated = [-x[D/2:], x[:D/2]]`）更直接的
做法是 **`data_move` 半区搬移 + 分段乘 -1**（负号融合进对应分段，零表零 pattern）。
`GatherMask` 指令与 `data_move` 搬移对 16bit 均可用；fp32 无 `GatherMask` 时优先搬移方案。

### 1.2 必须建表时：矢量生成，禁止标量 SetValue 循环

offset/索引表是规则行模式（如 `(row*D + 2i)*elemBytes`）时：
- ❌ 标量循环逐元素 SetValue（4096 次标量写 ≈ 17.7µs，标量管道串行）；
- ✅ 先少量标量写种子段，再用矢量加法**倍增扩展**（每趟把已填区按行距
  偏移翻倍），建表成本降到矢量指令级；
- ✅ 建表量按**本核实际有效行数**截断，不要按 tile 上限建满。

### 1.3 广播/重复操作数：用 masked repeat-stride 矢量运算直接消费

广播源（如 cos/sin 沿 S 维共享）**不要物化成中间张量**再算：
- 用带 repeat-stride 的掩码矢量乘加直接消费，源侧 repeat stride 置 0
  即"同一行重复用"；
- 行段游走（row-run）：把 tile 按"源行相同的连续行段"切分，每段一次
  矢量运算，避免逐行/逐元素处理。

### 1.4 输出直接算成最终布局

计算时就按输出的最终内存布局摆放中间结果（利用目标侧 repeat stride），
让写回退化为**单次连续 DataCopyPad**。反面结构：先算成交织/分段布局、
再多次跨步写回拼形状——写回次数翻倍且每段不连续。

### 1.5 广播加载：只搬去重后的源行

广播场景一个 tile 内多行共享同一源行：
- ❌ 逐行各搬一次（实测 64 行小拷贝 ~150µs @ R=4096）；
- ❌ host 侧 `expand().contiguous()`（S 倍拷贝，实测慢 3 倍）；
- ✅ 按 tile 覆盖的源行范围**连续搬去重后的行**（行数 ≤ validRows/S + 1），
  配合 §1.3 的行段游走消费。

### 1.6 dtype 能力分路（按 dtype 分类实现）

重排/搬运类的结构选择强依赖 dtype，按 dtype 分类单独实现：

| dtype | `GatherMask` 专用指令（硬件 pattern） | 搬移/重排（`data_move` 半区搬移、cat 式重排） | 建表 `Gather` |
|---|---|---|---|
| fp16/bf16（16bit） | ✅ stride-2 pattern（16bit 元素对） | ✅ 通用 | 不必要 |
| fp32 | ⚠️ 指令层支持 float（CANN dav_c220 断言、asc-devkit 性能表均含 float），但 stride-2 pattern 实测行为不符预期 → 优先搬移方案 | ✅ 通用 | 兜底 |

- `GatherMask` dtype 支持：stride-2 pattern 在 16bit 元素对上是主流路径；`float` 在指令层受支持
  （CANN 8.3.RC1 dav_c220 断言、asc-devkit 性能表均含 float），但 **fp32 上 stride-2 pattern 实测
  行为不符预期**——故 fp32 优先走 `data_move` 搬移/重排方案，而非硬凑 pattern
- **`data_move` 搬移等重排方案对所有 dtype 通用**（RoPE half 的搬移方案不挑 dtype）
- 按 dtype 分类单独实现，编译期 `if constexpr` 分派，不硬凑

> ⚠️ 术语区分：**`GatherMask` 专用指令（硬件 pattern）≠ `Gather` + 手工 mask 表（反模式）**。
> 后者在 half 上崩溃（mask 符号扩展垃圾、offset 构造错误、旋转失效），切勿混用。

### 1.7 Gather 兜底铁律（确实绕不开 Gather：不规则 gather/scatter、fp32 无 pattern 兜底）

确实要直接用 `Gather` 时，两条实测铁律（也是 §3「Gather+mask 表在 half 崩溃」的根因拆解）：

1. **源必须走独立 VECCALC，禁止 alias `TQue<VECIN/VECOUT>` 队列 tensor**：`DeQue` 出的队列 tensor 在
   `BUFFER_NUM>=2` 时会**跨迭代复用 slot**，alias 作 Gather 源会被下一轮 `CopyIn` 覆盖 → 多核非确定性
   （单核对/多核错，D 越大越易触发）。先 `DataCopy`/`Cast` 到独立 `TBuf<VECCALC>` 再 Gather。
   **禁止用 `BUFFER_NUM=1` 规避**——那会废掉双缓冲流水（实测大 shape 加速比因此跌到 0.18x）。
2. **`srcOffset` 取值须保证偏移后不超 UB 范围**：`Gather` 的 `count` **无"64 元素/8 DataBlock"上限**——
   官方 rotary rope 单次 `Gather(..., count=numHeads*headDim)`（典型 1024 fp32）即反证。offset 表用
   int32/uint32、**按字节缩放**（参考官方 `SetGatherSrcOffset`）；"越界"根因是 offset 构造错误
   （未按字节缩放 / 符号扩展），不是 count 本身。

> 一句话：§3 的 half 崩溃不是 Gather 本身的问题，而是「offset 构造错误 + 源 alias」；修掉这两条后
> Gather 能跑对，但仍比 `GatherMask`/`data_move` 慢，故仅作兜底。

## 2. host tiling 生成规则：核数分档

AIV block 分发存在 ~60-100ns/核的 ramp，小/中规模任务满核 launch 白付
1-3µs 分发倾斜。生成 tiling 时按规模分档：

| 规模带 | 核数策略 |
|---|---|
| 极小（行数 < 数十） | 按行数给核（1 行/核），宁少勿多 |
| 中（延迟敏感带） | 按"行数/每核行数配额"上取整，并夹在 [4, 16] 区间 |
| 大（吞吐带） | 满核 |

档位边界应在目标硬件上实测确定（固定核数 × 规模阶梯，多次取均），不要拍脑袋；
同一规则通常对 dtype 与广播/非广播同时成立。

## 3. 已知低效结构（生成时避免）

| 结构 | 实测结果 | 原因 |
|---|---|---|
| host 预计算 offset 表、经 GM 传入 kernel | 负优化（加缓存后仍低于基线） | 表只有几 KB，kernel 内矢量建表本就便宜；GM 传入 + 拷回 UB 新增搬运通道开销与屏障 |
| 为覆盖 half+interleave 引入 **`Gather` + 手工 mask/sign 表的统一方案** | half 上崩溃/旋转失效（长时间排查根因） | mask 符号扩展产生垃圾、`Gather` srcOffset 越界、half 旋转输出恒等；`GatherMask` 指令与 `Gather`+mask 表是两回事 |
| 为省同步删 V 管屏障 | 全阶梯差值 <±0.3µs（噪声内） | 屏障不是瓶颈，删了还引入正确性风险 |
| 为少分配而合并 InitBuffer | 预期 <0.3µs（< 测量噪声） | 不值得代码复杂度 |
| **克隆/搬运段微段化 + 段间 `PipeBarrier<PIPE_ALL>`**（如 512B 段 × 每段 2 次全屏障） | 等效带宽钉死 ~100GB/s（标杆 320~1556GB/s，10x 劣化） | 507035 的根因是 count 未 32B 对齐，与段长无关——微段化是误诊处方，每次全屏障都排空整条流水 |
| **串行 `TQue` depth=1**（确定性修复后的过矫正） | ~40% 性能损失 | 确定性来自 TQue 生命周期管理而非 depth=1；搬入/搬出严格串行、零重叠 |
| **同 stream 顺序 launch 间插 `stream.synchronize()`；host 侧 dtype 预处理 launch** | 小 case 地板翻倍（~22µs vs ~11µs）；设备侧 1.13x 被拖到端到端 0.83x | 同 stream 顺序 launch 天然有序，host sync 是纯开销；kernel 内可完成的 cast 外迁 = 白付 launch 固定开销 |
