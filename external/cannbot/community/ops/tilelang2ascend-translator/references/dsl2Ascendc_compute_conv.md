## Conv 计算详细参考：img2col + GEMM 与 LoadData3D 随路展开

本文档总结卷积类算子（conv2d）的 AscendC 实现与性能优化知识。
概览与判断规则见 `@references/dsl2Ascendc.md`，Cube 通用组织见
`@references/dsl2Ascendc_compute_cube.md`。

---

## 第一章：总体架构 —— host-zero-prep 多级 kernel 流水

卷积不要写成单个 kernel，也不要在 host 侧用 at:: 算子做预处理。推荐的形态是 **host 只发射 kernel + 分配 workspace，全部
布局变换由 device kernel 完成**：

```
wrepack (AIV)   : 权重 [Co, Ci, kH, kW] -> K-major ws [K, CoPad]（或 fused k 序）
im2col  (AIV)   : 特征图 -> im2col 矩阵 [totalRows, K]   ← 可被 P5 fused 模式消除
conv    (AIC)   : GEMM, GM --Nd2Nz--> L1 --LoadData3D--> L0 --Mmad(fp32)--> L0C --Fixpipe--> GM
epilogue(AIV)   : acc fp32 [totalRows, Co] + bias -> cast -> 目标布局输出
```

要点：

- 每级 kernel 都是独立发射，kernel 间通过 GM workspace 衔接；host 侧无任何 device 算子。
- 权重 repack 结果是（weight 内容, shape, dtype）的确定函数，可在 host 做 **LRU 缓存**
  （key = data_ptr + sizes + dtype + version，条目内持有 weight 强引用防止地址复用），
  命中时跳过整个 wrepack 发射。推理场景权重不变，这是净赚。
- AIC:AIV 混合核用 `KERNEL_TYPE_MIX_AIC_1_2`，AIV 侧用 `GetSubBlockIdx()` 切子核。

---

## 第二章：GEMM 映射的三种模式

同一个 conv kernel 按 case 特征选择 GEMM 组织方式（host 判定，传 mode 标量）：

| 模式 | A（M×K） | B（K×N） | 适用 | 省去 |
|:---|:---|:---|:---|:---|
| normal | im2col GM 矩阵 [rows, K]，k 序 (kh,kw,ci) | K-major 权重 ws | 通用回退 | — |
| direct 1x1 | 原始权重 [Co, Ci]（**翻转 GEMM**，跨图共享） | 原始 x NCHW [Ci, H*W]（本身就是 [K,M] ND 布局） | kH=kW=1, s=1, p=0, Ci%C0==0, (H*W)%C0==0 | wrepack + im2col 两级发射 |
| fused img2col | L1 常驻 NC1HWC0 特征图，img2col LoadData3D **随路展开**直接进 L0A | K-major 权重 ws（fused k 序，按 nIdx 常驻 L1） | Ci%C0==0（whole-fmap 路径用真实 Ci 不吃 pad 通道）、xNz+B 面板（物理分配）≤ 480KB | im2col GM 物化（最大单收益） |

> **C0 = 32B / 元素宽**：fp16/bf16 → 16，fp32 → 8（NZ/ZN 分形 `[16 行, C0 列]`）。
> 表中所有 %16 类约束的实质都是 %C0——现行实现 gate 硬编码 16 是因为
> fused/direct 路径只路由 fp16/bf16；窗口化 band 路径（xtrans_pad 上片到
> CiPad）不受 Ci 对齐限制，任意 Ci 可进。

direct 1x1 的关键认知：1x1 卷积的 im2col 是恒等变换，NCHW 原始布局的 x 已经是
`[K=Ci, M=H*W]` 的 GEMM B 操作数，权重做 A 即可，acc = [Co, H*W]，epilogue 按
通道行加 bias 直接写出 NCHW，全程零 transpose。

> **fp32 输入另有精度适配维度**（3-chain 已退役 → bf16 六项拆分 → 原生 fp32
> cube 定稿，见第八章），不改变本表的 A/B 组织，但 fp32 cube 的 BASE_K/C0 不同。

---

## 第三章：数据通路 API 选型

| 通路 | API | 实测结论 |
|:---|:---|:---|
| GM(ND) → L1(NZ) | `DataCopy` + `Nd2NzParams` | **ND 数据进 L1 的唯一可靠通路**（限定：GM 侧为 ND 行主序——raw torch 输入 / 自产 AIV kernel workspace 属此类，本包全部 GM workspace 即如此）。plain 2D DataCopy GM↔L1 在 dav-c220 不可靠（L1 侧得到非分形字节，cube 无法正确消费，实测踩坑）；UB-relay 替代实测 net negative。**若上游已给 NZ**（FE 场景 FMAP/Weight 直接 NZ 落 GM，A2/A3 常见），禁止再 Nd2Nz，走 fractal 保形拷贝；Nd2Nz 有随路格式转换开销，设计期优先让生产者直接产 NZ、全链路避免 ND |
| L1(NZ) → L0A/L0B | `LoadData` + `LoadData3DParamsV2` | 普通 tile 用 l1H=1/filter=1 退化为纯搬运；conv 用 img2col 模式（filter/stride/dilation/pad 硬件处理） |
| L0C → GM | `Fixpipe` + `FixpipeParamsV220` | fp32 NZ → ND；写大矩阵时 `dstStride` 必须是完整行宽，不是 tile 宽 |
| GM ↔ UB（非对齐） | `DataCopyPad` | 见第六章对齐陷阱 |
| UB 内 16×16 转置 | `TransDataTo5HD` | **c220 不支持 bf16**：`ReinterpretCast<uint16_t>` 按位转置，全平台安全 |

Mmad/L0 组织沿用 matmul 成熟实践：每个 K_L1 迭代只 `DeQue` 一次、内层传偏移指针、
全部内层迭代完成后只 `FreeTensor` 一次；m/n 尾块按满 baseM/baseN 计算，Fixpipe 只
写实际范围（stale 行列丢弃），不要为尾块缩小 Mmad（m 侧缩减实测产出 garbage）。
尾块的整体策略（host padding 禁令与 kernel 内尾块标准形态）见第九章。

---

## 第四章：fused img2col LoadData3D 随路展开（最大优化）

### 原理

im2col 物化路径的流量是 `特征图 -> UB -> GM im2col -> L1 -> L0`，其中 GM 往返
是主要耗时来源（实测占比最大）。fused 模式让 **A 侧不落地**：

```
xT GM [img, H*W, Ci] --Nd2Nz--> L1 NC1HWC0 常驻 --img2col LoadData3D--> L0A Zz tile
                                  （pad/stride/dilation 硬件随路处理）
```

- B 侧复用 normal 路径的 K-major 权重 ws，按 nIdx 常驻 L1 深度 1。
- CopyOut / epilogue / m/n 分解与 normal 路径完全一致，改动只集中在 A 侧加载。
- NCHW 输入需要先由一个 AIV xtrans kernel 生成 xT `[H*W, Ci]`；NHWC 输入本身即
  channel-inner，**直接用原始 x，零中转**。

### 关键实测发现：硬件 img2col 展开的 k 序是 (ci1, kh, kw, ci0)

**C0 通道块在最外层**，与软件惯用的 (kh, kw, ci) 不同。权重 ws 必须按同一 k 序
repack（wrepack 增加 fused 布局输出，缓存键带上布局标志）。两种 k 序在 Ci==16 或
kH*kW==1 时恰好重合——如果第一个验证 case 恰好是 Ci=16，会"蒙对"，换 case 即暴露，
务必用 one-hot 探针（单元素置 1 观察落点）实测确认，不要凭文档假设。

---

## 第五章：有效优化手段清单（按实测收益排序）

| # | 优化 | 实测收益 | 机制 |
|:---|:---|:---|:---|
| 1 | fused img2col 消除 GM 物化 | 收益最大（消除 GM 往返） | 见第四章 |
| 2 | host 侧 weight repack LRU 缓存 | 省去整个 wrepack 发射 | 推理权重不变，结果确定可缓存 |
| 3 | AIV kernel 内同步点合并 | im2col 耗时大幅降低 | 所有 (kh,cib) slab 先一次性加载、共享**一个** MTE2→V 同步点，再背靠背转置（V pipe 程序序保证）；替代每个 slab 2 个 flag pair |
| 4 | TransDataTo5HD repeat 批量转置 | xtrans 耗时显著下降 | 每个 ci block 一次调用、repeat 走遍 16×16 子块；标量逐项构造 list（每子块 32 个 list entry）是瓶颈 |
| 5 | direct 1x1 flipped GEMM | 省两级 kernel 发射 | 见第二章 |
| 6 | epilogue 跨行大块处理 | epilogue 耗时显著下降 | 整块连续区域一次 load + 行内标量 Adds（V pipe 行间免同步）+ 一次 cast + 一次 store；替代逐行 flag pair |
| 7 | depth-2 A1/B1 跨 (blk,panel) 预取 | conv 耗时明显下降 | 当前块计算时发射下一块的 GM→L1；B 槽只分配实际列数（≤baseN）使两块面板能同时放进 L1 |
| 8 | stagger-aware 核数调优 | conv 耗时小幅下降 | MIX kernel 发射 stagger 与核数成正比，**满 wave 的少核 > 带残尾 wave 的最大核数**：waves=ceil(blocks/aic)，dim=ceil(blocks/waves)；小 grid 按 ~2 tile/核封项 |
| 9 | n 自适应 L0B/Mmad | 窄 N 时 cube 工作量显著减少 | nActual≥64 时按 nActual 加载；**nActual<64 保持满 baseN**（nActual 过窄会触发 L0C 读写冲突，实测） |

---

## 第六章：dav-c220 硬件行为实测要点

1. **img2col LoadData3D k 序 = (ci1, kh, kw, ci0)**（C0 块最外层）——见第四章。
2. **TransDataTo5HD 不支持 bf16**：按 uint16_t 位级转置。
3. **plain DataCopy 要求 32B 倍数传输**：fp32 load 需 len%8==0，fp16/bf16 store 需
   len%16==0；不满足会**静默截断**（实测非对齐 store 被截短），非对齐一律
   DataCopyPad。
4. **DataCopyPad 非 32B 倍数的 blockLen**：UB 侧行 pitch 会向上取整到整 32B block
   （两个方向都取整），寻址必须按取整后的 pitch；且非 32B 行走慢速 dummy-pad 路径，
   大量非对齐行时考虑改算法而不是硬拷。
5. **MIX kernel 发射 stagger 与核数成正比**：核数选择按满 wave 原则（第五章 #8）。
6. **LoadData3D Zn 的 channelSize 非 C0 倍数会跳过部分分形块**（fp16/bf16
   下 C0=16，kernel 代码即 round16）：空间尾必须按 C0 对齐，否则对应列数据损坏。
7. **重复利用同一 UB buffer 的跨 pipe 同步**：下一轮 MTE2 写 vs 上一轮 V 读、下一轮
   V 写 vs 上一轮 MTE3 读，都要显式 SetFlag/WaitFlag；**最后一次迭代可省略回同步**
   （flag 往返是纯延迟，无事可重叠）。
8. **MTE3 store 与同 buffer 后续 V 计算**：store 读 trans buffer，下一轮转置覆写它，
   MTE3 必须 drain 后 V 才能继续。
9. **img2col LoadData3D 几何 gate（现行实测）**：`kExtension > 64` 挂死；
   `channelSize=16` 走慢路径（pad 到 ≥32 规避）；窗口化 band 路径仅验证
   「方形核 ≤5×5、sh==sw、dh==dw、CiPadWin≤768」的几何 envelope，出
   envelope 一律回退 normal。早期"通道块索引 C1≥15（Ci≥240）产垃圾"的
   记录来自 whole-fmap 常驻时代，现行窗口化实现已无此 gate——勿照搬旧
   gate，也不要凭"archive 同结构"推断安全域。
10. **AIV 侧无 `HardEvent::MTE2_MTE3`**（AIC-only），误用 → 507014 死锁；AIV 等价
    保护用 SetFlag/WaitFlag 旗标对。
11. **aicore timeout 的 kernel 会使该 NPU 进入持久坏状态**（同设备 torch_npu 原生
    conv 都输出错误）——调试崩溃类问题必须先换干净设备对照，避免把设备污染误判
    为代码回归。
12. **V 管异步写 vs MTE2/MTE3 读（及 V 读 vs MTE2 写）的 WAR 竞态必须立即配对
    V_MTE2 / MTE3_V**：手工 buffer 复用链（split-K 求和、批处理 im2col 批次边界）
    不受"上一 op 的 EnQue/DeQue 已覆盖"保护，各踩过一次精度/覆写坑。
13. **GM→UB DataCopyPad dstStride=0 的多行打包要求行块长本身 32B 对齐**（fp32
    ln%8==0），否则行起点未对齐、后续向量读出错——成批加载门限必须按此设定。
14. **Level-0 count+repeat 形式的向量 API（Cast/Add 等），count 不得超过单条指令
    元素宽度**（fp32=64，fp16=128）：超出部分**静默保留陈旧值，无报错**，坏点呈
    指令宽度周期分布（误差分布探针可一次定位）。
15. **PipeBarrier<PIPE_ALL> 的适用边界（实测）**：PIPE_ALL 会泄流相邻迭代
    在飞的 MTE3 存储（曾致显著性能回退）——**纯 V/MTE 配对的流水永远不需要
    它**：主 cube-GEMM 流水与向量化 AIV kernel（conv/im2col/wrepack/xtrans/
    epilogue）全程零 PIPE_ALL，跨 pipe 同步由 TQue EnQue/DeQue + 最窄
    SetFlag/WaitFlag 旗标对覆盖。**例外（现包 4 个 kernel 实测在用）**：
    ① 标量引擎主导的 kernel 的 V↔S 跨引擎交接——vector `Duplicate` 初始化
    → scalar `GetValue/SetValue` 累加 → vector `Adds` 读出（direct_conv 注释
    "make the cross-engine handoff explicit"、nanrepair 同形态）；
    ② `TransDataTo5HD` 前后边界（转置引擎管道归属跨管道，prep5mf）；
    ③ 调试标记写入后的跨核等待（oneshot）。判据：主数据路径被标量引擎主导、
    或一条同步链跨 ≥3 条管道时 PIPE_ALL 才是合理选择（且这些路径都是兜底/
    低性能敏感路径）；可向量化的主路径一律最窄旗标对，个别单管道屏障
    用 `PipeBarrier<PIPE_V>`（代价可忽略），不升级到 ALL。
16. **TransDataTo5HD per-call 成本是固定值**（与 rows、stride 几何无关）
    ——小 co 转置存在结构性耗时地板，只能靠减少调用次数（coC 最大化、
    hi/lo 打包单次调用）或更换转置机制突破。
17. **单文件多 `extern "C"` kernel 的注册风险需实测确认**：历史场景双 kernel
    单文件在 rtRegisterAllKernel 阶段报 107000（binary parse failed）；但
    现行 conv2d_base 实现单文件多 kernel（最多 4 个）正常工作——该结论
    依赖编译/打包链版本，上车前用最小用例验证，既不盲目拆文件也不盲目合文件。
18. **DataCopyPad 每次调用有固定成本、每 2D block 有描述符开销**：小 case
    kernel 设计必须最小化调用次数，整块（blockCount=1）大传输优先；预载类任务
    "更多并行更优"是伪直觉——whole-copy 预载体积与任务数无关，任务越多 L2
    冗余越大（多 band 对比实测）。
19. **x 内容 LRU 缓存对 benchmark 稳态有效**（harness 输入张量只搬一次）：staging/
    预处理结果可按 (data_ptr+version+sizes) 缓存，正确性依赖 version_counter
    追踪原地改写 + 条目持有强引用防 data_ptr 复用（wrepack 缓存同款机制）。
20. **UB 侧元素粒度位移不可行，唯一可行模式是 GM scratch 位移副本 + blockCount=1
    整块预载**：UB 内三类位移手段全部受限（向量 src 需 32B 对齐、UB→UB 拷贝
    32B 块步长、GatherMask src 需 32B 对齐）——需要 {-1,0,+1} 这类元素级窗口
    位移时，在 GM scratch 写好多份位移副本，再整块大传输载入。

---

## 第七章：tiling 与资源预算经验值（910B）

| 参数 | 取值 | 依据 |
|:---|:---|:---|
| L1 预算（A1/B1 depth-2 面板） | 448 KB | 910B L1=512KB 留 headroom（对齐碎片、NZ padding、编译插桩） |
| fused gate（xNz + 常驻 B 面板） | 480 KB | 实测 gate：`xNzBytes + bResBytes ≤ 480KB`，物理分配口径 |
| baseM / baseN / baseK | 128 / 256 / 64 | L0C 单 tile 128×256×4B = 128KB 占满 L0C；**fp32 cube 用 baseK=32**（L0A/L0B 字节容量折半） |
| C0 | 16（fp16/bf16）；8（fp32） | 分形 C0 = 32B / 元素宽 |
| UB 预算 | 192 KB 总量留 headroom | im2col slab / epilogue acc tile 按此反推行数 |
| epilogue acc tile | ~16K fp32 元素/tile（64KB UB） | rowsPerTile = 8192/CoPad 反推 |

> **平台适用范围（已核查）**：本表全部经验值仅适用 **Ascend910B（dav-c220 /
> NPU 2201）**——L1=512KB、L0C=128KB、UB=192KB 均为 2201 口径。
> 448KB 预算经多轮实测跑通（fused xNz 常驻 + A1/B1 depth-2 双面板 + 常驻
> B 面板共存 L1）；480KB 的 fused gate 与 TileLang 侧
> `conv2d-template.md` 的 `FUSED_L1_BUDGET` 同口径。

---

## 第八章：fp32 精度适配路线（演进定稿）

fp32 conv 的精度适配经历三代演进，**禁止把中间方案当定稿**：

| 代 | 方案 | 结论 |
|:---|:---|:---|
| fp16 hi/lo 3-chain | 拆分 hi+lo，3 条 GEMM，残差解析上界约 2⁻²² | **退役**：相消 case 上达不到 fp32 精度门禁 |
| bf16 hi/mid/lo 六项拆分 | 保留 hi·hi+hi·mid+hi·lo+mid·hi+mid·mid+lo·hi | 解析残差低于 fp32 eps；代价是 6 条 GEMM 固定成本 |
| **原生 fp32 cube 单 pass** | 严格 fp32 Mmad | **定稿默认**：无拆分误差、无额外 GEMM |

### 原生 fp32 cube 落地要点（三件套 + 两条禁令）

- 分形 C0=8、BASE_K=32（L0A/L0B 按字节容量折半，2×[128,32]×4B=32KB/queue）。
- Mmad m/n/k 均 16 对齐 + `kDirectionAlign = true`；`SetHF32Mode(false)`
  关闭 CANN 默认 HF32 舍入。
- **禁令一（L0A Fmatrix）**：`SetFmatrix(1, m, PAD, FMATRIX_LEFT)` 须在
  **每次** LoadData3D 前重设——Fmatrix 是 sticky 状态，上一次 RIGHT 会
  污染本次加载。
- **L0B fractal 转置路径**：B 按 [k,n] 存储时
  `SetFmatrix(1, l1W, PAD, FMATRIX_RIGHT)` + LoadData3D `fMatrixCtrl=true`。
- **禁令二（M-pipe→Fixpipe）**：用 `TQue<CO1>` 的 EnQue/DeQue 串行（可靠），
  **禁止手搓 barrier**——漏补即 Cube/Fixpipe 竞态死锁、AICore 挂死并污染
  设备。CO2 双 CO1 slot 让 L0C drain 与 M-pipe 乒乓，避免 Fixpipe 排空停滞。
- 长 K 链的累加误差用 accLanes 拆多条独立累加链（精度手段，**不是性能
  split-K**；单链长度有实测安全界）。

### 已知风险与兜底

- **窄小图（窄 W、低行数）fp32 cube 链有 aicore crash 史**（MTE read
  on-chip buffer out of bounds）——小 shape envelope 内改走**纯 AIV 直卷**
  （单发、无 LoadData3D、无 L0/L1 queue、无跨核同步，按构造避挂）。
- **fp32 链式 kernel 有 case 顺序触发的 aicore-timeout 死锁史**——链式
  kernel 的验证必须覆盖 case 顺序排列，单 case 验证通过不等于链安全。
- **±inf 语义**：fp32→bf16 拆分时 inf 的低位分量为 NaN，含 inf 的接收域
  累积为 NaN，与参考 fp32 conv 的 NaN 模式一致；host 侧**禁止**用 device
  reduction 做有限性探测（评测对 aclrtMemcpy 计数有约束），有限性问题靠
  路由兜底而非 host 探测。

### bf16x6 六项拆分的 k 序契约

权重按 split 布局 repack（hi/mid/lo 三段），是与 fused 序
`(ci1,kh,kw,ci0)`、normal 序 `(kh,kw,ci)` 并列的**第三种 layout**——
wrepack 缓存键必须带布局标志，三种 layout 不可混用、不可跨路径复用。

---

## 第九章：多轮性能迭代策略总结（fp32 conv 实测）

fp32 Conv2d 全参数（groups/dilation/stride/bias）vs torch_npu 融合 conv
标杆，多 case 精简集，`--quick` 口径。经多轮迭代 geomean 加速比显著
爬升（各轮数值由性能环节归档，不在知识文档枚举），迭代轨迹按策略主题
记录如下。

> **范围声明**：本章 R1/R3/R4/R5 轨迹来自 archive conv2d_base 的多轮迭代。
> ~~R2（depthwise）与 R6（stride-2 去交织）两条专用路径~~已删除——它们属带
> groups 参数的卷积任务经验，无 groups 参数的 conv2d 从未落地（如需引用见
> evolution-strategies 的 C-5/X22 路由与 archive 任务原始记录）。

### 轮次轨迹

| 轮 | 策略主题 | 核心改动 |
|:---|:---|:---|
| R1 | fused img2col 落地 | 4 个 fused kernel（xtrans/wrepack/gemm/epilogue）+ host 按 case 路由，normal 路径保留回退 |
| R3 | 小 case 流水优化 | xtrans 列表提升 / k1-direct / 工作区 LRU / split-K / im2col 成批 |
| R4 | 测量口径修正 | 非代码改动：op_summary 全量采集口径失真，统一 `--quick` |
| R5 | 瓶颈排序逐项击破 | epilogue 合并转置 / im2col band 拆分 / flat-span / k1direct 单 kernel |

### R1 fused img2col（机制见第四章）

- 硬件 img2col k 序 (ci1,kh,kw,ci0) 用 one-hot 探针实测确认，勿凭文档假设。
- C1≥15 硬件缺陷 gate（第六章 #9，whole-fmap 常驻时代记录，现行窗口化
  实现已无此 gate）；2D UB→GM 批量存储（srcStride=0）实测有效。
- 教训：fused 收益集中在消 GM 物化，但对小行/微行 case 无效，需后续轮次补齐。

### R3 小 case 流水优化

- **先分解后动手**：msprof 逐 kernel 分解推翻"gemm 流水链主瓶颈"预设——真凶是
  epilogue_fused 与 xtrans，gemm 耗时占比远低于预期。
- TransDataTo5HD 固定成本定律（第六章 #16）→ xtrans 指针列表提升出任务循环
  （耗时显著下降）+ host 侧 T 选择避开碎尾 tile。
- host 工作区（im2col/acc/xtrans buffer）按 shape **LRU 缓存**，消除每调用
  at::empty/zeros + 清零 kernel（缓存零值语义等价）。
- split-K 收益有限：gemm per-block 固定成本主导而非 kt 循环——**先探针量化固定成本再决定是否拆 K**。
- im2col 8 行成批门槛 ln%8==0（第六章 #13）。
- k1-direct 初版（hilo 纯拆分 kernel 走 normal flip GEMM）已证 1x1 免转置可行，
  R5 Opt-D 将其推到单 kernel 形态。

### R4 测量口径（方法论铁律）

- **msprof 全量 aic-metrics 采集（op_summary 模式）会使 asc 侧虚高、ref 侧
  虚低**（同一二进制下同一 case 可测出完全相反的两个结论）：
  对比测试必须 `--quick`（1 轮采集、无 aic-metrics）。
- 冷启动 host 开销是假象：稳态 host≈device，每次 launch 有固定开销。
- 台间噪声明显：多轮测量趋势一致才可采信，单轮绝对值不可作为回归判据。

### R5 瓶颈排序逐项击破

| 优化 | 机制 | 实测效果 |
|:---|:---|:---|
| Opt-A epilogue 合并转置 | hi/lo 打包单次 TransDataTo5HD（repeatCount=2*nb 连续两段）+ 跨步 Cast/Add 重组 + host R=256 候选（UB 门限） | 大通道 case epi 耗时显著下降 |
| Opt-B im2col oh-band 拆分 | 任务粒度 (n,g,ci,kh)→(…,oh-band)，band 数按 2×aicNum 自适应 | im2col 耗时大幅下降，个别 case 接近标杆 |
| Opt-C flat-span 小空间路径 | W==Wo 时 col 行 = 平面连续平移拷贝，1 次装载 + 边缘 kw mask（Duplicate+标量置零）+ 成对存储；同步用最窄 V_S/S_V 旗标（#15） | 小空间 case im2col 耗时数量级下降，接近标杆 |
| Opt-D k1direct 单 kernel | 1x1/s1/p0 新增纯 AIV Axpy kernel：免 GEMM/免 wrepack/免 hi-lo/免转置，3-launch→1-launch | 1x1 case 加速比大幅提升 |

### 贯穿各轮的方法论

1. **分解先行**：每轮先做逐 kernel 分解/探针实验再动代码（两次推翻预设
   瓶颈）；多 case 精度失败先看误差分布（周期坏点=指令宽度问题），勿逐个猜。
2. **固定成本建模**：小 case 由 per-call/per-block 固定成本主导（转置固定
   成本、launch 固定开销、gemm per-block 固定成本）——先建实测成本模型，
   再选优化点；预期收益未量化前不上车（无效 barrier 削减、split-K 低收益均为教训）。
3. **单变量迭代**：每项改动独立"精度全过 → quick 实测无回归 → 保留/回退"，
   无效项回退并记录走偏点，不堆积未验证改动。
4. **专用路径 > 通用打磨**：1x1/小空间等专用 case 的收益全部来自专用
   kernel 或专用路径，通用 im2col+GEMM 的渐进打磨收益递减。**专用路径的
   gate 必须先用全量集验证不误伤其他 case**（窄 gate 收紧到仅个别 case
   命中才有净收益）。
5. **风险预算**：有设备污染史的方向（**全链 3-launch MIX 融合**，CrossCore
   死锁 + 设备污染史，探针本身危及设备）宁可放弃；**但 conv+epilogue 两级
   MIX 已有可靠同步范式**（`CrossCoreSetFlag` + `PIPE_FIX` 序释放对端
   AIV，convepi 家族），不在放弃之列。调试崩溃先换干净设备对照（第六章 #11）。
6. **tiling 参数单点计算、显式下发**：host 与 kernel 各自推导同一 tiling 参数是
   高危模式，任一侧引入新候选都会静默破坏另一侧 buffer 预算（曾致 UB 超限
   崩溃）。

---

## 第十章：尾块处理策略 —— host 不 padding 到整块再 crop（opt8 审计定稿）

### 策略

**默认不要在 host 侧 padding 到整块再 crop。优先在 kernel 内做尾块处理；host
侧 padding 只在"后端路径确实无法表达 valid 范围"时作为受控 fallback，且必须把
额外分配与拷贝计入成本。**

理由：

1. **纯开销转移，不减少计算**：host padding 只是把 padded-copy 从 kernel 搬到
   host——kernel 仍按完整 block 算一个 tile，多出来的是一次
   `ceil(M/block)*block*K` 的 workspace 分配、一次 GM 读+写的真实数据拷贝
   （不是 metadata 视图）、以及输出侧 padded 存储和无效行写回。kernel 内做法
   只是把尾块 tile 的无效行在 UB/L1 内填 pad_value，不产生 GM 流量，且可与
   搬运重叠。
2. **语义风险**：归约/归一化类算子中 padding 行会进入分母或统计量——必须用
   幺元填充（sum 填 0、max 填 -inf、min 填 +inf）或 mask，否则有效行结果错误；
   分母必须用逻辑长度（如 rstd 的除数是逻辑 D），不能用 padded 长度。M 只是
   batch/独立行维度时 padding 行数值无害，但仍属纯浪费。
3. **不改善并行度**：M < block 时 task 数仍是 `ceil(N/block)`（乘 batch），
   padding 不会增加任务数。此时 Cube 有效利用率就是 M/block，应缩小 block、
   沿 N/batch 分核，或 M 极小时改向量/GEMV 路径，而不是 padding。

### 推荐做法

- **能整除就整除**：优先按指定 case 选合法 block；可在 Python factory 按余数
  类生成专用 kernel，主路径保持静态整 tile。
- **仍有尾块**：
  - GM 侧只访问 valid 范围——精确 slice 或 DataCopyPad / T.copy 的 pad_value；
  - UB/L1 按完整 tile footprint 分配，无效 lane 填幺元或计算 mask；
  - 输出只回写有效矩形——裁剪发生在 kernel 的写回 extent 里（Fixpipe
    mSize=mActual / 动态切片 `Y[m:m+valid_m]`），不是先写 padded 输出再由
    host 切片。
- **GEMM 的 K 尾块**：A/B 的 K 片段补 0（0 对累加是幺元）；M/N 尾块只影响被
  丢弃的行列。Mmad 尾块按满 baseM/baseN 计算、Fixpipe 只写实际范围（见第三章），
  不要为尾块缩小 Mmad。

### 标准形态对照（审计结论）

对 conv2d_base 做 host padding 专项审计，覆盖 host 插件 + 全部 kernel 路径，
结论**零可消除点**——现行实现即标准形态，可作新算子的对照模板：

- **host 插件**：y 全部路径精确形状裸 return，无 view/slice；workspace
  全部由 kernel 从 raw x_ptr 填充；零 aclrtMemcpy；mImg/totalRows 全程
  精确不取整；LRU 缓存含**权重 repack**（内容是 weight/shape/dtype 的
  确定函数，key 带 k 序布局标志）与 **workspace**（按 shape，消除每调用
  分配+清零）；x 内容缓存对 benchmark 稳态有效（第六章 #19）。
- **预处理 kernel**：xtrans NO-PAD；xtrans_pad 家族是 PAD-IN-WS——CiPad 零列
  在 workspace 内由 kernel Duplicate 上片（为 LoadData3D 16 通道粒度 + K-chunk
  64 的 GEMM 结构配对所必需），无 GM pad 拷贝；im2col halo 零填充为语义必需。
- **conv/epilogue 写回**：所有 y store 裁剪 valid 矩形；LoadData3D padValue=0
  上片；Fixpipe mSize=mActual。

两处已知无害标记（fastAny pad-Co 垃圾列喂 discarded acc 列、32B 粒度过读）
经评估不值得承担共享热头文件布局涟漪风险，有意保留——审计时注意区分"目标点"
（GM 侧 padded 拷贝、无效行写回、超量 DataCopyPad extent）与"无害标记"。

### 受控 fallback 条款

仅当后端路径确实无法表达 valid 范围（如某 DMA 描述符格式要求整行传输且
kernel 内无 pad 机制）时才允许 host padding fallback，且必须：

1. 在 design / 实现文档中显式声明 fallback 触发条件；
2. 把额外 workspace 分配与 GM 拷贝计入性能成本，与 kernel 内方案对比后再决策。