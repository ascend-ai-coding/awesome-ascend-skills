# AscendC 排序 / TopK / 采样类算子实现指南（Sort / TopK / TopKTopP 通用）

> 适用范围：`sort / topk / top_k_top_p / sampling` 等**找第 k 大 / 排序列选 / 概率阈值
> 过滤**类算子的 AscendC **生成与实现**。全部条目来自实测，只给规则与量级——
> 生成时按此写，一次到位。设计阶段决策见 designer references/sort_topk_design.md。

## 0. 生成结构：值域二分 vs 分段 sort 双路径

「找第 k 大」按规模分野为两条**完全独立**的路径，在 device 侧 Init（buffer 分派）与
Process（计算入口）最早处分支，两条路径的 buffer 互不污染：

| 条件 | 路径 | 结构 |
|---|---|---|
| N ≤ 阈值（实测 910B3 交叉点 ≈8192） | 值域二分 | O(iter·N) 多遍全量扫描，免排序 |
| N > 阈值 | 分段 sort | 单遍分段硬件 sort + 归并取 top-k |

> 🛑 选路前先过设计层否决（sort_topk_design §1.0）：输出为**有序值/有序索引**
> （argsort 下标、gather 按序消费）→ 值域二分不适用（只产阈值/掩码），排序路径；
> 分组小 N（每组 ≤1024）→ 排序路径；k≤2 的 reduce-max 替代须先做硬件效率估算。
> 复杂度估算（`iter×N×3op` vs 单遍+归并固定开销）写入设计文档后才允许选二分。

> 值域二分的性能档位由**迭代次数 × N 遍扫描**决定；分段 sort 由**单遍 + 归并固定开销**
> 决定。二分的迭代次数下界必须按「**含 padding 的实际 range**」算（`n > log2(range/gap)`），
> 不能只看真实数据 range——padding 值过大会把二分精度档位拉低，导致 kth_value 附近
> **误收集**（精度坑）。用 eps 提前收敛替代固定次数，对 range 变化更健壮。

## 1. 值域二分实现铁律（全部来自实测事故）

| # | 铁律 | 违反后果 |
|---|---|---|
| 1 | **迭代次数下界按含 padding 的 range 算**：`PAD_VALUE` 若设太大（如 -1e6），`FindMinMax` 的 range 变 ~1e6，`range/2^iter` 大于真实 gap，kth_value 附近误收集 | 精度退化（bf16 边界元素误收/漏收） |
| 2 | **exp 预计算一次，TopP 二分搬 exp 域**：`exp(x-max)` 只依赖输入和 max，与阈值无关，却易被重算 40×N 次（超越函数极贵）；应一次算好存 workspace，TopP 二分读预计算值累加 | 40×N 次超越函数，V 管减负失败 |
| 3 | **每遍扫描 = 分批 CompareScalar + Select + ReduceSum 计数**（矢量），禁止标量逐元素计数 | 标量管道串行，慢一个数量级 |
| 4 | **-inf 掩码值必须真 -inf（内联表达式）**：`static const float NEG_INF = -1.0f/0.0f` 会被 AICORE 编译器折叠成 0.0，mask 输出错 | 输出 0 而非 -inf |
| 5 | **二分搜索端点用有限值填充**：端点用 -inf 时 `-inf >= -inf` 恒真，二分卡死；用有限值（如 -1e6，但受 #1 约束） | 二分不收敛 |

## 2. 分段 sort 实现铁律（照抄官方 `top_k_top_p_sample` / `top_k_v3` 结构）

1. **分段硬件 Sort + 归并取 top-k**：每 2048 段 `Sort`（value+index 交错降序），
   再用 `MrgSort4` 归并取 top-k；后续块用 `CompareScalar(GT)` + `GatherMask` 原地收集
   候选（> 当前 kth 的才入队）再归并。
2. **`Concat` 在 910B3 是空操作**（`concat = src`）：`SortOneTime` 直接
   `Sort(dst, src, idx, tmp, len/32)`，无需先 Concat。
3. **取负 trick 实现升序**：`Sort` 只支持降序，对 `-value` 降序即得 value 升序
   （取负是精确符号位翻转），再取负恢复。
4. **`Sort` 的 tmp buffer 需要 `2 × dataLen` 大小**；`repeatTime = dataLen/32`。
5. **常数名避开系统头**：`THIRTY_TWO` 等与 `topk_common_utils.h` 匿名命名空间撞名，
   改用自有命名（如 `SORT32_UNIT`）。

## 3. GatherMask mask 模式铁律（507035 vector core exception 根因）

`GatherMask` mask 模式（`reduceMode=true`，`vreducev2`）是分段 sort 收集候选的
标准手段，但有三条硬约束，违反直接 vector core exception（507035，plog 报
`The UB address accessed by the VEC instruction is not aligned`）：

1. **`src0` 与 `src1Pattern`(mask) 必须在不同 buffer**：mask 与 src0 同 buffer 触发
   对齐违例。收集时 mask 用一块 buffer、src0 用另一块。
2. **`dst` 必须原地（`dst=src0`，无偏移）**：跨 block 偏移累加（`dst = srcData[m]`）
   踩 `vreducev2` 对 dst 偏移/对齐的约束。正确做法：每 block 原地 GatherMask 压缩，
   再标量/矢量回填到累加 buffer（幸存元素总数 ≤ 上限，标量开销可忽略）。
3. **910B3 默认 `VERSION_V2` 是 dav_c220 支持的**：`GatherMaskCal` 的
   `ASCENDC_REPORT_NOT_SUPPORT(mode == VERSION_V2, ...)` 语义是「非 V2 才 abort」
   （宏为 `if (!(cond)) raise`），勿误读成"V2 不支持"。

## 4. tie 语义对齐（位置阈值 → 值阈值）

参考语义（`sort(ascending, stable)` + `mask < kth`）是**值阈值**：保留所有
`value >= kth_value`（并列全留，数量可 > k）。分段 sort 天然是位置阈值（恰好 k 个），
**必须两步对齐**：

```
分段 sort → 读 kth_value（排序结果第 k 个值）
         → 向量化重收集所有 `value >= kth_value`（CompareScalar(GE) + GatherMask 原地压缩）
         → 收集结果接「取负升序 sort + softmax + 升序 cumsum 位置阈值 + scatter 回原序」
```

> 不做这步，bf16（7bit 尾数、tie 密集）的 top-k 边界会欠掩码/过掩码，match_ratio 掉到
> 0.97（fp16 0.996、fp32 1.0）。dtype 尾数越少 tie 越密，越必须对齐。

## 5. 采样类评测约束（运行经验）

1. **性能对比标杆 = 官方融合算子**（`torch_npu.npu_top_k_top_p` / aclnn），**不是
   torch eager 拼接实现**（`sort→gather→masked_fill→softmax→cumsum→scatter_` 9 个独立
   kernel）：用 torch eager 作标杆会把性能目标拉虚高/失真。精度 golden 仍是 `model.py`，
   官方算子仅作交叉验证。这是**评测标杆口径**在排序/TopK 类上的具体实例，通用规则见
   translator SKILL.md 关键限制。
2. **p ∈ [0,1]**：采样类算子的 p 是概率阈值。workload 生成时 p 若用默认 randn
   （标准正态），约 40~70% 的 p 超 [0,1]，官方算子在 p>1 有 bug，评测无法对齐。
   **必须给 p 加 `range: [0,1]`（生成 uniform）**。
3. **k ≤ min(N, 1024)**：k>N 或 k>1024 会导致官方算子 aicore exception（507015）。
4. **计时红线 + 后台跑测**：性能只接受 device 侧 msprof 时间；长跑用 `setsid ... &
   < /dev/null` 后台 + 轮询存活（nohup 会被 bash 工具超时 SIGTERM 误杀）。
5. **小 B 小 N 档位**：`blockDim=B` 时小 B 只用 1~2 核、固定开销主导（实测 2×256 仅
   0.5~1.1x），此类 case 属已知弱项，记录即可，不作为阻塞。
