# Scan / Cum 模板集成指南

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 0. `.template` → `.h`

本目录 `.template` 文件是参考模板源码。使用前**去掉 `.template` 后缀**转成同名 `.h`（如 `cum_streaming_scan.template` → `cum_streaming_scan.h`），拷贝到项目 `op_kernel/`（保留 `dav310/` 子目录结构），再 `#include`。

对应关系：

| template | → 头文件 | 用途 |
|----------|---------|------|
| `scan_tiling_data.template` | `scan_tiling_data.h` | packed TilingData 定义（host/device 共用） |
| `scan_tiling.template` | `scan_tiling.h` | host 侧流式 tiling 选型 |
| `dav310/scan_base.template` | `scan_base.h` | 促进类型 / CastTrait / log2 / GetCacheID |
| `dav310/cum_streaming_scan.template` | `cum_streaming_scan.h` | 流式扫描（含 argmin/argmax）。**仅作参考实现，勿用于性能路径**，见 §1.1 |
| `dav310/cum_tile_resident_scan.template` | `cum_tile_resident_scan.h` | **N>1 性能路径**：tile 常驻扫描 + ping-pong 双缓冲，含索引 |
| `dav310/cum_row_kogge_stone.template` | `cum_row_kogge_stone.h` | N==1 的保底路径：行内 Kogge-Stone + 行批处理 + ping-pong 双缓冲，含索引 |
| `dav310/cum_oneway_sklansky.template` | `cum_oneway_sklansky.h` | 长单行 R 的对数深度扫描（Cumsum/Cumprod） |
| `dav310/cum_lane_parallel_scan.template` | `cum_lane_parallel_scan.h` | **N==1 性能路径**：Gather 虚拟转置 + lane 并行顺序扫描（含两行交错 LaneParallelScanPair），消掉 log2(R) |
| `dav310/cum_core_partition_scan.template` | `cum_core_partition_scan.h` | 1D 超长 R 跨核借 R + 保序进位（含完整 ProcessBorrowR/ReduceRowRange/ReadCarry/WriteSegTotal/InitCarry） |

`cum_row_kogge_stone.template` 是**类体内粘贴区**（整段包在 `#if 0` 里），
按注释里列的成员依赖（`xGm_/valGm_/idxGm_`、三个 TBuf、`idxWorkOff_`、`tl_`）
拷进自己的扫描类即可，不要当成独立可编译单元。内含两部分：
- **Kogge-Stone 扫描**（`ScanRowRange`/`KsInit`/`KsLevel`/`KsApplyCarry`/`KsCastOut`）
- **行批处理 + ping-pong 双缓冲**（`ScanRowBatchLoop`/`BatchCopyIn`/`BatchCompute`/`BatchCopyOut`），
  需要额外成员变量 `gsS_`/`xRowStride_`/`vRowStride_`/`oRowStride_`/`xBufElems_`/`valBufElems_`/`outBufElems_`/`evMte2V_[2]`/`evVMte3_[2]`/`evMte3V_[2]`/`evMte3Mte2_[2]`

## 1.1 选型：先看数据怎么进 UB，再看用什么扫描算法（cann-bench Cummin 20 用例实测）

扫描类算子的性能瓶颈几乎不在"用哪种扫描算法"，而在**搬运的粒度与描述符数量**；等这一项摆平之后，
瓶颈才轮到**向量指令条数**。cann-bench Cummin 20 用例上逐项优化的实测轨迹：

| 改动 | 综合得分 | 平均加速比 |
|------|---------|-----------|
| 逐 r 流式扫描（起点） | 54.89 | 0.13x |
| tile 常驻 + 行内 Kogge-Stone | 66.98 | 0.58x |
| int64 索引在 UB 内 Cast 后直写 | 68.93 | 0.70x |
| 跨核借 R（修 1D 超长用例） | 69.29 | 0.71x |
| fp16/bf16 原生 16 位比较 | 69.84 | 0.74x |
| 行批处理 + 双缓冲（N==1） | 73.37 | 0.91x |
| 借 R 合成单次 launch（SyncAll） | 73.75 | 0.93x |
| **Gather 虚拟转置 + lane 并行顺序扫描** | 79.38 | 1.59x |
| 两行交错填延迟 | 79.95 | 1.69x |
| **16 位输入改用 fp32 比较类型** | 81.63 | 1.85x |
| **mode 0（N>1）加 ping-pong 双缓冲** | 82.41 | 1.92x |
| 借 R 的 phase B 复用行内扫描 | 82.59 | 1.93x |

前五步在搬运层，后面几步在计算层——**lane 并行扫描一项就吃掉了 0.66x**，是全程最大的单点收益。

> 注意倒数第三行**推翻了第五行**：`fp16/bf16 原生 16 位比较` 是在 Kogge-Stone 时代测出的收益，
> 换成 lane 并行扫描后完全反转（见 §2 的更正）。**换扫描算法后必须重测比较类型的选型。**

### 第一层：数据怎么进 UB

按 R 轴是否连续（融合成 `(M, R, N)` 后看 N）分两条路：

| 形状 | 正确做法 | 反面做法与代价 |
|------|----------|----------------|
| `N > 1`（dim 非最后一维） | **tile 常驻**：每行一次 `DataCopyPad(blockCount = rLen)` 把整段 R 搬进 UB，布局 `[R][lane]`；扫描沿 R 顺序推进，running(min,idx) **常驻寄存器**，循环体内零 DMA | 逐 r 搬入搬出：`2*R` 次小搬运 + 每 r 三个 `SetFlag/WaitFlag` 完全串行，实测慢 **8~50x** |
| `N == 1`（dim=-1，R 在 GM 上连续） | **一次 `DataCopyPad` 搬 k 整行**（`blockCount=k`），UB 内走 §1.2 的 lane 并行顺序扫描 | 沿用 N>1 的 lane 折叠（借 M 轴）：每行要发 R 个 `blockLen = elemBytes` 的描述符，且 32B 对齐把每元素撑成一个 32B 槽位（fp16 浪费 16x、fp32 浪费 8x）。实测 0.19x → 换 Kogge-Stone 后 0.32x |
| `M*N` 远小于核数且 R 很长（典型 1D） | **跨核借 R**，见下 | 只按 work-item 分核 → 单核跑完，`R=1000003` 实测 0.01x |

选型口诀：**能一次搬连续大块就绝不逐点搬**；lane 维度只在 N 本身够宽时才用 N，N==1 时改用"沿 R 的对数深度扫描"而不是把 M 折进 lane。

### 第二层：向量指令条数

搬运摆平后测到的现象很说明问题——**每秒处理的元素数与 dtype 无关**（int32/fp32/bf16 都是约 3.5e10 elem/s）。
既然 bf16 只搬一半的字节却不变快，说明已不是带宽受限，而是受限于向量指令条数；而 fp16/bf16 被提升到 fp32 后
每 vreg 仍只有 64 lane，与 fp32 完全一样，提升本身就是那道天花板。两个可落地的动作：

- **16 位原生比较**：`C = T` 直接比，每 vreg 128 lane，且进出各省一趟 Cast。代价是索引仍是 int32（64 lane），
  一个值寄存器要配 2 个索引寄存器，比较掩码需 `UnPack<LOWEST/HIGHEST>` 拆成两半分别驱动索引侧 `Select`。
  在 Kogge-Stone 路径上实测大 R 的 16 位用例快 15~21%。

  > **⚠️ 这条结论只对 Kogge-Stone 成立，走 lane 并行扫描时必须反过来（一律升 fp32）。**
  > 16 位的 `Gather`/`Scatter` 没有全速路径。同为 `1152×1024` 个元素的实测：
  >
  > | | Kogge-Stone | lane 并行 |
  > |---|---|---|
  > | fp16（原生 16 位比较） | 18.55µs | 15.05µs |
  > | fp32 | 28.58µs | **9.27µs** |
  >
  > Kogge-Stone 下 fp16 比 fp32 快 1.54 倍（符合 128 vs 64 lane 的预期），
  > lane 并行下却慢 1.62 倍。注意 fp16 的 gather 指令数只有 fp32 的一半（36 vs 68）、
  > 元素触达数也相当，却慢近两倍——**方向反了就只能是 gather 本身的位宽路径问题**。
  > 改成一律升 fp32 后 fp16 降到 9.39µs，与 fp32 打平；6 个 16 位用例快 25~38%。
  > 排查时先用 `ResourceConflictRatio` 排除了 bank 冲突（`bank_cflt` 仅 0.1%）。
- **把嵌套 Select 折成一个掩码**：`isNaN(B) ? B : (B<=A ? B : A)` 等价于 `(le | nan) ? B : A`，
  用 `Reg::Or(takeB, leMask, nanMask, mk)` 合并后，值和索引各省一条 `Select`，16 位路径还省一半 `UnPack`。

先量一量再动手：把 `KsLevel` 里索引侧的向量操作整段注释掉跑一次，就能知道值/索引各占多少。
本例测出索引只占约 25%，据此判断"加宽值侧"值得做、"把索引也压成 int16"不值得。

### 第三层：换掉 log2(R) —— Gather 虚拟转置 + lane 并行顺序扫描

前两层把搬运和指令宽度都摆平之后，剩下的大头是 Kogge-Stone 自带的 **log2(R) 倍工作量放大**：
`R=8192` 要做 13 级，每级都是全长的 Compare/Select。而顺序扫描只需 `O(R)`，
问题是顺序扫描在 UB 里天然只有一条 lane 在动。

破法是**不做真转置也能让 lane 并行**：把一行切成 VLC 段（VLC = 一个 vreg 的 lane 数），
让 lane `j` 负责第 `j` 段，用 `Gather` 以 `gIdx = lane*S + t` 的索引一次取回
"每段的第 t 个元素"，等价于免费拿到了转置后的一行。三个相位：

1. 每条 lane 沿自己那段顺序归约，得到段内 `(min, idx)`；
2. VLC 个段结果正好塞满一个 vreg，用**寄存器间 `Gather`**（`Arange(-d)` 做 lane 平移）
   跑 log2(VLC) 级 Kogge-Stone 得到段间 exclusive 前缀——log 因子从 log2(R) 缩到 log2(VLC)；
3. 带着前缀重扫一遍，`Scatter` 就地写回。

总工作量 `2R + log2(VLC)`，`R=8192` 下相对 13 级的 Kogge-Stone 少了约 6 倍算力。
实测 case4/6（`8192×8192`）从约 1780 µs 降到约 770 µs。

四个必须注意的点：

- **三个相位放同一个 `__VEC_SCOPE__`**：段结果和 lane 间前缀全程留在寄存器。
  拆成多个 scope（每级回写 UB）实测**倒退 19%**。
- **`LocalMemBar` 必须写在 scope 内部**：放到外面会直接触发
  `device error type 3, error code 507035`。
- **索引与比较类型必须等宽**，否则要拆掩码，收益被吃光。由于比较类型一律是 4 字节（见 §2 的更正），
  索引统一用 int32、出口一步 Cast 到 int64 即可；`int16 → int64` 一步 Cast 编译得过但结果错
  （下标远超 R），必须逐级 2 倍展宽——升 fp32 后这个坑连带消失了。
- **段长必须防 bank 冲突**：gather 的 stride 是 `S * sizeof(C)`，一旦是 16B 的倍数，
  所有 lane 落到同一 bank，实测 case3 慢 **5.5 倍**。把 `S` 强制取奇数即可绕开。

行数够时再叠一层**两行交错**：同一个 scope 里跑两行的相位一，两条独立依赖链互相填
Compare→Select 的延迟槽，实测再快约 3%。

### 第四层：三条流水的重叠 —— 用「占比之和」判断该不该上双缓冲

判据只有一句：**`PipeUtilization` 里 `vec + scalar + mte2 + mte3` 四个 ratio 加起来
≈100% 就是完全串行，必须上 ping-pong；加到 130% 以上才算真重叠。**
这比逐条算理论耗时快得多，也不会被 L2 命中率误导。

本例 `N > 1` 的 tile 常驻路径（六个用例）加双缓冲前后：

| case | 加前 | 四条占比之和 | 加后 | 之和 | 变化 |
|------|------|------------|------|------|------|
| 8 | 18.57µs | 101% | 13.94µs | 132% | −25% |
| 17 | 13.68µs | 103% | 9.49µs | 155% | −31% |
| 19 | 15.95µs | 102% | 11.00µs | 168% | −31% |
| 14 | 73.21µs | 113% | 64.79µs | 173% | −12% |
| 10 | 500.7µs | 102% | 492.9µs | 170% | −2% |
| 5 | 2697µs | 101% | 2690µs | 170% | −0.3% |

case 5/10 几乎没动：它们加双缓冲后 `mte3` 单项就占到 74~80%，已经是**纯写带宽受限**
（int64 索引占总写量的 4/5）。**双缓冲只能把串行变重叠，不能突破最长那条流水**——
所以先看占比之和决定要不要做，再看最长单项决定能做到哪。

两个落地要点：

- **`rChunk` 会因此减半**（缓冲翻倍），块数随之加倍，这正是流水需要的，不必回调。
  host 侧的 `bytesPerR` 公式要同步乘 2，否则 UB 溢出。
- **不要为省 UB 让两个缓冲区叠用**。本例原先把 int32 扫描区叠在 int64 输出区的高半段上，
  靠 Cast「读指针 4B/元素、写指针 8B/元素」的错位就地展宽。加双缓冲时这会在
  搬出该输出区的同时被下一块的扫描写坏，**必须拆成独立分配**。
- 扫描沿 R 的 **carry 依赖是串行的**，所以双缓冲重叠的是「搬入下一块 / 算当前块 / 搬出上一块」，
  不是计算与计算。carry 留在 UB 上跨块续传，都在 V 流水内，程序序即依赖序，不需要额外 flag。

### 跨核借 R（先 2 次 launch 跑通，再合成单次 launch）

`M*N` 喂不饱核时把 R 切成 P 段、每段一个核，分两阶段：

1. **phase A**（mode 3）：各核只读自己那段求出段内合并结果 `(val, idx)` 写进 workspace，**不写输出**；
2. **phase B**（mode 4）：各核读回前序所有段的结果合成进位，带着进位完整扫一遍自己那段并写出。

段间屏障就是两次 launch 在同一条 stream 上的顺序语义，比核内 `SyncAll` 稳，**先用它跑通**。

跑通之后再合成**单次 launch + `AscendC::SyncAll()`**：一次 launch 的固定开销约 9 µs，
在这种小用例上和整个搬运时间同量级，实测 `R=1000003` 从 35.7 µs 降到 19.2 µs。
两个坑：`SyncAll` 前要 `PipeBarrier<PIPE_ALL>()` 确保段结果已落 GM；
**轮空的核也必须走到 `SyncAll`**，判空要写成 `bool active = ...` 而不是 `if (...) return`，
否则死锁。

另外 phase A 只需要段内的 `(min, argmin)`，那是**归约**而不是扫描，没有 log 因子。
复用扫描代码再把前缀丢掉会白付 log 倍算力，改成单趟归约实测省 38%。
**别让 phase A 顺手把局部前缀写出去**：那样 phase B 要把整个输出回读再写一遍，访存量从 `x + 输出`
涨到 `x + 2*输出`；让 phase A 只读、phase B 重扫一遍，多花的是算力而不是带宽，实测 `R=1000003`
从 2002 µs 降到 59 µs（0.01x → 0.22x）。

两个实现细节：各段在 workspace 里**独占一个 32B 槽位**，避免相邻核写进同一个 32B 块；
段数要用"先由段数定段长、再用段长回推段数"收敛一次，否则尾部会有分不到数据的空核。

### 试过但不划算：SIMT

`dim=-1` 形状上曾试过 SIMT 标量路径（`cce::async_invoke`，绕开 UB 与搬运描述符），
实测比 Kogge-Stone **慢 5~8 倍**，1D 超长用例慢近 100 倍——标量直访 GM 没有 UB 聚合，
在这类数据量下完全打不过向量路径。官方 cummin 给 SIMT 设的门槛是 `N < 128 && M > 12800 && R < 12800`，
即**行极多且每行极短**，本用例集无一命中。选 SIMT 前先拿这个门槛卡一下。

### 计时方法：本地测量有 ~30 µs 的地板

torch 自定义算子经 python 派发，`n=64` 与 `n=1000003` 都测得 30 µs——小 kernel 的真实
耗时完全被派发开销吞掉。做微优化 A/B 时要么用 cann-bench 的设备侧计时（一轮约 6 分钟），
要么**把 M 放大 8~10 倍**再测：每行的 R/dtype/dim 不变，每行的开销特征就不变，比值可信。
本轮一度因为忽略这个地板，得出「借 R 的算法改造无收益」的错误结论。

## 1. 入口写法（`<<<>>>` 单翻译单元工程，参考 softmaxNewKernel/cumminNewKernel）

host 的 `main` 与核函数写在同一个 `.asc`，`main` 里直接 `NAME<<<blockDim, nullptr, stream>>>(...)` 调用。用 `__NPU_ARCH__` 把 arch35 引入与入口体限定在 device pass，host pass 只见入口空壳签名。

```cpp
#define CUMMIN_ENTRY(NAME, TX, TP)                                                   \
  extern "C" __global__ __aicore__ __vector__ void NAME(                             \
      GM_ADDR x, GM_ADDR values, GM_ADDR indices, ScanTilingData tiling) {           \
    TPipe pipe; ScanOps::CumStreamingScan<TX, TP> op;                                \
    op.Init(x, values, indices, &tiling, &pipe); op.Process();                       \
  }
CUMMIN_ENTRY(cummin_fp32, float, float)
CUMMIN_ENTRY(cummin_fp16, half, float)
CUMMIN_ENTRY(cummin_bf16, bfloat16_t, float)
CUMMIN_ENTRY(cummin_i32, int32_t, int32_t)
```

host 侧：`platform_ascendc` 取 UB 字节与 AIV 核数 → `scan_tiling::TilingScan(shape, plat, td)` 得 `blockDim` 与 `ScanTilingData` → 按 dtype dispatch launch。

## 2. 常见编译问题（遇 Reg API 报错时按序尝试，禁止直接降级为高层 API）

- **命名空间与构建配置**：CANN 9.0+ 模板统一使用 `AscendC::Reg::`。若 `Reg` 或 `RegTensor` 未声明，先核对 `--npu-arch=dav-3510`、`kernel_operator.h` include 链和同工程实际 flags；禁止手工添加别名。
- **`CMPMODE` 归属**：`CMPMODE` 在全局 `AscendC` 命名空间，不在 `AscendC::Reg` 下。写 `CMPMODE::GT` 而非 `AscendC::Reg::CMPMODE::GT`（本模板已按此写）。
- **`Select` 语义**：`Select(dst, src0, src1, mask)` → mask 为真取 src0，否则取 src1（对照 `kernel_reg_compute_vec_cmpsel_impl.h` 的 `vsel`）。
- **API 签名适配**：对照 `$ASCEND_HOME_PATH/x86_64-linux/asc/include/basic_api/reg_compute/` 实际头文件调整参数。

## 3. 数值与并列语义自检

- fp16/bf16 **能原生比较就原生比较**（见 §1.1 第二层）；只有需要在标量侧比较进位时才升 fp32，
  因为编译器不支持 16 位浮点的标量类型转换（`(float)bfloat16_t` 直接报 `not support bf16 type cast`）。
  升降转用 `DIST_UNPACK_B16` / `DIST_PACK_B32`。原生路径要构造 `+inf` 幺元时同样不能用标量转换，
  直接给位模式：fp16 `0x7C00`、bf16 `0x7F80`。
- 把 cummin 的语义收敛成一个**幺半群**再实现，顺序扫描与并行扫描才会给出同样的结果：

  ```
  combine(A_靠前段, B_靠后段) = takeB ? B : A,   takeB = isNaN(vB) || (vB <= vA)
  ```

  等价于"NaN 视作最小、并列取更晚下标"的 min，满足结合律与幂等，因此可安全用于 Kogge-Stone / Sklansky。
- 并列（tie）语义对齐 torch.cummin/cummax = **last occurrence**，故用 `LE`（`vB <= vA` 时取靠后的 B）。
- NaN 必须显式粘滞：只写 `Compare` 不够。cann-bench 的 `value_range=["nan","nan"]` 生成的是
  **NaN 与实数混合**的输入；若只用 `GT` 判 `x > curMin`，因 `(实数 > NaN) = false` 会把 NaN "恢复"成实数。
  正解是再加一路 `Compare<NE>(x, x)` 判 NaN，与 `LE` 的结果 `Or` 成一个掩码：`(le | nan) ? x : curMin`。
  （实测：只用 `GT` 时 20 用例中 case 14 精度失败。）
- **索引直接产出 int64**：torch 的 indices 是 int64，扫描过程用 int32（省一半寄存器与带宽），
  搬出前在 UB 内 `Cast` 成 int64 再 `DataCopyPad`，不要落 int32 到 GM 再由 plugin 转换（多一整趟读写）。
  int32 工作区与 int64 输出区可以复用同一块 buffer：`Cast` 按元素升序推进、写指针（8B/元素）
  始终落后于读指针（4B/元素），只要工作区偏移 ≥ 待转换元素数就不会读到已被覆写的数据。
  **该偏移必须按 vreg（256B）对齐**，否则 R 不是 8 的倍数时工作区上的向量访存会触发
  `device error type 3, error code 507035`。

## 4. 多 work-item 复用 UB 的 WAR 屏障（务必保留）

一核处理多 work-item / 多 R 分块时，UB 缓冲被反复复用，需要**两个方向**的屏障，缺一不可：

- **MTE3 → V**：本 item 末次 `CopyOut(MTE3)` 还在读 running-(min,idx)，下一 item `InitRunning(V)` 就覆写它们。
  `MTE3_MTE2` 只约束 MTE3→MTE2，**不约束 MTE3→V**。必须在 `ProcessWorkItem` 结尾补 `MTE3→V`，
  否则每个 work-item 最后一个 r 的 idx 被覆写成 0（值因走独立 valOut 缓冲而正确，掩盖问题，
  仅在"一核多 item + 大 R"时暴露）。
- **V → MTE2（经 MTE3 传递）**：本块的 V 还没读完输入缓冲，下一块的 `CopyIn(MTE2)` 就可能抢先覆写它。
  补一个 `MTE3→MTE2` 即可，靠 `V →(V_MTE3)→ MTE3 →(MTE3_MTE2)→ MTE2` 的传递序兜住。
  实测踩坑：单 work-item（M=1）时只有一次迭代，测试全过；一旦 M>1 一核跑多行就出现数据相关的零星错值。

## 5. Kogge-Stone 沿 R 扫描的两个坑

- **分块必须从高地址往低地址处理**。第 k 级要算 `val[i] = combine(val[i-d], val[i])`，原地更新时
  低地址块是高地址块的读源；降序处理才能保证读到的是本级之前的值。
- **跨 rChunk 的进位不能只靠前置哨兵**。在缓冲区前放 GUARD 个哨兵、填成上一块的 carry 看似能让进位自动
  并进来，但下标为 `2^k-1` 的元素在 `d >= GUARD` 的各级不再被更新（`i < d`），其窗口恰好触到 0 而不会
  落进哨兵区，因此拿不到 carry。正解是哨兵只填**幺元**（`+inf` / `INT_MAX`），跨块进位另用一趟显式
  pass 合并（相对 log2(R) 趟只增约 8%）。实测踩坑：只有 `R > rChunk` 的用例（`R=1000003`）会失败。
- **哨兵长度取 1 个 vreg（256 字节）而不是固定元素数**。比较类型是 16 位时那就是 128 个元素，
  这样数据区起点恰好落在 vreg 边界上；按元素数写死会让 16 位路径的数据区错位。
  buffer 备量按最宽的那种比较类型算，两种实例化才能共用同一块 UB。

## 6. 模板接入约束

### 坑 1（致命）：`AllocEventID` vs `FetchEventID`

ping-pong 双缓冲的事件 ID **必须用 `AllocEventID`**，不能用 `FetchEventID`。
`FetchEventID` 返回该事件类型的「默认」ID，每次调用返回同一个值，导致 ping-pong
两个 buffer 的事件互相 alias——`Wait(buf[0])` 吃掉 `Set(buf[1])` 的 flag，数据在
搬运中被覆写，**不报错，只出非确定性精度错误**。

`AllocEventID` 从事件池分配唯一 ID，ping-pong 多 buffer 必须用它。

### 坑 2（致命）：`ScanRowRange` 缺少 lane 并行路径

`ScanRowRange`（mode 1/3/4 的核心函数）必须包含 lane 并行条件分支：
`gsS > 0 → LaneParallelScan` 和 `gsS == 0 → KoggeStone` 两条分支。
若只保留 Kogge-Stone，case 4/6（R=8192, mode 1）走纯 Kogge-Stone 的
`log2(R)=13` 趟扫描，耗时 1786µs；走 lane 并行只要 770µs，差 2.3 倍。

### 坑 3：`Init()` 方法不可遗漏

模板已提供 `Init()`（含 `InitBuffer` 分配 UB + `AllocEventID` 分配事件 ID）。
使用时**必须调用** `Init()`，否则 UB 未分配、事件 ID 未分配，运行时崩溃。

### 坑 4：Host 侧 `CalcTiling()` 缺失

使用 `scan_cummin_tiling.h` 的 `CalcTiling()`（含 6 种 mode 选择 + UB 预算自适应 +
跨核借 R 段数计算）。不要自行编写 tiling 计算，除非有明确的定制需求。

### 坑 5：`CalcMode5MFold` k=0 回退

`CalcMode5MFold` 中 UB 预算不够时 `k` 返回 0，`CalcTiling` 自动回退到 mode 1。
**不要强制 `k = 1`**——单行仍超 UB 时会触发 `device error 507035`。
