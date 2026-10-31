# 行内 Kogge-Stone 扫描（N == 1 / dim=-1 主力）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

对应模板：[dav310/cum_row_kogge_stone.template](dav310/cum_row_kogge_stone.template)

## 什么时候用

融合成 `(M, R, N)` 后 **N == 1**，也就是扫描轴落在最后一维、在 GM 上连续。
这是 `dim=-1` 的常态，cann-bench Cummin 20 个用例里有 13 个是这种形状。

不要在这种形状上沿用 `N > 1` 的 lane 折叠路径。折叠路径把 M 行折进向量道，
每行要发 `R` 个 `blockLen = elemBytes` 的搬运描述符；再加上 32B 对齐会把每个元素
撑成一个 32B 槽位（fp16 浪费 16x、fp32 浪费 8x），实测只有 0.19x。

## 流程

```
for 每个 rChunk：
  1  DataCopyPad          整段连续搬进 UB，1 个描述符
  2  KsInit               哨兵区填幺元 + 转成比较类型 C + Arange 生成绝对下标
  3  KsLevel × log2(rLen) 第 k 级把 val[i] 与 val[i-2^k] 合并
  4  KsApplyCarry         把上一块的进位显式并进本块每个元素（仅非首块）
  5  取本块末元素作为下一块的 carry
  6  KsCastOut（C != T 时）+ Cast 索引到 int64
  7  DataCopyPad × 2      values 与 indices 搬出
```

关键点：整个 `log2(R)` 级扫描发生在 UB 内，循环体里没有任何 DMA。
每级的分块之间互相独立，ILP 等于分块数（`rChunk / VLC`）。

## 四个必须踩对的点

### 1. 分块必须从高地址往低地址处理

第 k 级要算 `val[i] = combine(val[i-d], val[i])`，原地更新时低地址块是高地址块的读源。
降序处理才能保证读到的是本级之前的值。模板里体现为
`i0 = i0Min + (chunks - 1 - c) * VLC`。

### 2. 跨 rChunk 的进位不能只靠前置哨兵

在缓冲区前放哨兵、填成上一块的 carry 看似能让进位自动并进来，但下标为 `2^k - 1`
的元素在 `d >= VLC` 的各级不再被更新（`i < d`），其窗口恰好触到 0 而不会落进哨兵区，
因此拿不到 carry。

正解：哨兵只填**幺元**，跨块进位另用一趟显式 pass（`KsApplyCarry`）合并。
相对 `log2(R)` 趟只增约 8%。实测踩坑：只有 `R > rChunk` 的用例会失败。

### 3. 哨兵长度取 1 个 vreg（256 字节）而不是固定元素数

比较类型是 16 位时那就是 128 个元素。这样数据区起点恰好落在 vreg 边界上；
按元素数写死会让 16 位路径的数据区错位。buffer 备量按最宽的比较类型算，
两种实例化才能共用同一块 UB。

### 4. 一核处理多行时要补两个方向的 WAR 屏障

- `MTE3 → V`：本行的 CopyOut 还在读 UB，下一行的 `KsInit` 就覆写它。
- `MTE3 → MTE2`：本行的 V 还没读完输入缓冲，下一行的 CopyIn 就抢先覆写。
  靠 `V →(V_MTE3)→ MTE3 →(MTE3_MTE2)→ MTE2` 的传递序兜住。

单行（M=1）时只有一次迭代，测试全过；一旦一核跑多行就出现数据相关的零星错值。

## 16 位原生比较

搬运摆平之后，瓶颈从带宽转到向量指令条数——实测每秒处理的元素数与 dtype 无关
（int32/fp32/bf16 都是约 3.5e10 elem/s）。bf16 只搬一半的字节却不变快，
说明 fp16/bf16 提升到 fp32 后每 vreg 仍只有 64 lane 就是那道天花板。

把比较类型 `C` 取成 `T` 即可原生比较，每 vreg 128 lane，进出各省一趟 Cast。
代价是索引仍是 int32（64 lane），一个值寄存器要配 `NI = 2` 个索引寄存器，
比较掩码需 `UnPack<LOWEST/HIGHEST>` 拆成两半分别驱动索引侧的 `Select`。

实测大 R 的 16 位用例快 15~21%；`R` 小到只有一两个 chunk 时反而略慢，
拆掩码的固定开销摊不掉。所以这是个**跟 R 相关的选择**，不是无脑开。

约束：编译器不支持 16 位浮点的标量类型转换（`(float)bfloat16_t` 直接报
`not support bf16 type cast`）。所以凡是需要在标量侧比较进位的路径
（比如跨核借 R 的段间进位）只能继续用提升后的 fp32。构造 `+inf` 幺元时同样
不能用标量转换，直接给位模式：fp16 `0x7C00`、bf16 `0x7F80`。

## 把嵌套 Select 折成一个掩码

`isNaN(B) ? B : (B<=A ? B : A)` 等价于 `(le | nan) ? B : A`。
用 `Reg::Or(takeB, leMask, nanMask, mk)` 合并后，值和索引各省一条 `Select`，
16 位路径还省一半 `UnPack`。

## 索引直接产出 int64

torch 的 indices 是 int64。扫描过程用 int32（省一半寄存器与带宽），
搬出前在 UB 内 `Cast` 成 int64 再 `DataCopyPad`，不要落 int32 到 GM
再由 plugin 转换（那样多一整趟读写，实测 66.98 → 68.93 分）。

int32 工作区与 int64 输出区可以复用同一块 buffer：`Cast` 按元素升序推进、
写指针（8B/元素）始终落后于读指针（4B/元素），只要工作区偏移 ≥ 待转换元素数
就不会读到已被覆写的数据。**该偏移必须按 vreg（256B）对齐**，否则 R 不是 8 的倍数时
工作区上的向量访存会触发 `device error type 3, error code 507035`。

## 行批处理 + ping-pong 双缓冲（mode 5）

当整行装得下 UB（`R <= rChunkKs`）时，不要逐行处理——逐行时每行都要付一整套固定开销
（搬入/搬出共 3 个描述符、3~4 对 flag），R 小而 M 大的形状（典型 `[4097, 511]`）下
这部分能占到一半以上的时间。

模板中已补充 `ScanRowBatchLoop` / `BatchCopyIn` / `BatchCompute` / `BatchCopyOut` 四个函数，
提供完整的行批处理 + ping-pong 双缓冲骨架：

- **一次 `DataCopyPad(blockCount=k)`** 搬 k 行进 UB，固定开销摊薄 k 倍
- **x/val/out 各开两份 ping-pong**，4 组事件（`MTE2_V` / `V_MTE3` / `MTE3_V` / `MTE3_MTE2`）
  使"搬入下一批 / 算当前批 / 搬出上一批"三段重叠
- **末尾两批排空** MTE3_V / MTE3_MTE2（没有后继消费，留着会和下一轮等待错配）
- `BatchCompute` 内部按 `gsS_` 字段选择：`gsS_ > 0` 走 lane 并行（两行交错 `LaneParallelScanPair`），
  `gsS_ == 0` 走 Kogge-Stone（R 很小时 log2(VLC) 固定开销更划算）

不做双缓冲时四条流水占比之和只有 ~101%（完全串行），加上后升到 132%~173%，实测快 12~31%。
`k == 1` 也走这条路——批处理没了，但双缓冲带来的跨行重叠仍在（R 大时正是靠它）。

详见模板 `cum_row_kogge_stone.template` 的类体内粘贴区中 `ScanRowBatchLoop` 及其辅助函数。
