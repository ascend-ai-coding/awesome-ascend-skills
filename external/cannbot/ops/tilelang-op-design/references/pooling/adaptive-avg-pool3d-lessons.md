# Adaptive Pooling Lessons: adaptive_avg_pool3d 生成经验

> **定位**：前向 adaptive pooling（Avg/Max，3D/2D/1D）专属。窗口非固定，区别于固定 stride 的 [row-granularity.md](row-granularity.md)。
> 来源：adaptive_avg_pool3d 端到端（TileLang → AscendC）对比官方 `ops-nn/pooling/adaptive_avg_pool3d`，完整逐文件对比见任务目录 `comparison_official_vs_generated.md`。
>
> **⚠️ 数据口径（避免与 PR 描述混淆）**：本文出现三个不同口径的加速比，含义各异，引用时须注明——
> - **TileLang 基线 0.1262x**：Phase 3 Step 4 性能迭代基线（`perf_tuning/measure_baseline.py`），受 TileLang 0.1.4 Ascend 后端结构性限制（AUTO_SYNC 排空），不反映最终 AscendC 性能；
> - **AscendC 分 case**：常规窗口 case[1-5] >1.1x，`reduce_all (1,1,1)` 场景 ~0.15x（host 逐块搬运 + 原子合并开销 vs 官方预注册归约引擎）；
> - **AscendC 6-case geomean（v2，ops-profiling msprof）1.004x**：主指标，PR 描述表中「0.86 → 1.004」即此口径（0.86 为 v1 的 AscendC 6-case geomean）。

## 0. 一句话

自适应池化的语义 = 每输出点按 `start=floor(idx*isize/osize)`、`end=ceil((idx+1)*isize/osize)` 推导窗口（整数化公式见下），**NDHWC 布局 + C 向量化照用**，但窗口计算改为逐输出点推导而非固定行偏移。

## 1. 窗口索引：整数公式（与 PyTorch 语义 bit-exact 对齐）

```cpp
__aicore__ inline int32_t StartIndex(int32_t idx, int32_t osize, int32_t isize) {
    return (idx / osize) * isize + ((idx % osize) * isize) / osize;
}
__aicore__ inline int32_t EndIndex(int32_t idx, int32_t osize, int32_t isize) {
    return 1 + ((idx + 1) * isize - 1) / osize;
}
```

- 直接用官方 common.h 公式，**不要**用 `ceil((idx+1)*isize/osize)` 的浮点等价式——整数化避免尾差，6 场景 MERE=0。
- 窗口索引可在 kernel 内联算（整数运算开销可忽略，省一个 UB 索引缓冲），也可像官方批量预计算到 UB 查表（仅当要多窗口共享平面加载时才需要，见 §4）。

## 2. reduce_all / (1,1,1) 必须走专用快路径（性能坑）

**症状**：`output_size==(1,1,1)` 时一般路径对全空间归约做逐平面小块 DataCopyPad（wlen×cLen），MTE2 固定开销主导，实测 **0.105x**（136us vs 官方 14us）。

**官方做法**：op_api 层把 (1,1,1) 直接路由给 `l0op::ReduceMean`（框架预注册归约 op）+ viewCopy，**不进本算子 kernel**。

**教训**：
1. adaptive pooling 生成时，(1,1,1) 场景必须在 host 判定并走归约快路径，否则全量性能被它拉垮（geomean 里它是唯一 <1x 的）。
2. 若在 kernel 内做快路径（chunk 合并搬入），有两个必须验证的坑：
   - **`DataCopyPad blockCount > 256` 时 MTE2 只搬运前 256 个 block**，目标缓冲保留陈旧数据 → chunkRows 必须 ≤256。
   - **大块连续搬入在 multi-batch（n≥1）随机输入下读出错误数据**（实测偏差 1.2e-2）——多 batch 语义必须单独用随机输入验证，不能只看单 batch 用例。

## 3. 布局：NDHWC 对 adaptive pooling 依然适用（修正旧注）

`layout-strategy.md` 旧注「自适应 pooling 不适用」应理解为：**不适用的是「固定行偏移滑动窗口」计算方式**，NDHWC 布局 + C 向量化照用。本次 adaptive_avg_pool3d 就是 NDHWC（host permute）+ kernel 沿 C 向量化 + 逐输出点窗口推导，case[1-5] 全 >1.1x。

## 4. 多输出 W 点共享一次平面加载（官方 MULTI_W 思想，主要结构差距）

**问题**：朴素实现每输出点单独 `DataCopyPad(blockCount=wlen)` 只搬自己的窗口切片。相邻输出 W 点的窗口在输入上重叠，**逐点重读把 GM 读流量放大到 maxWindowWLength 倍**。upscale 场景（OD×OH×OW ≫ D×H×W）输出点多、窗口小，最吃亏。

**官方 MULTI_W**：一次 `DataCopyPad(blockCount=wend-wstart)` 搬入整段 `[wstart, wend)`（从本输出点窗口起点到窗口覆盖的输入最大范围），UB 中同时累加 `windowWNum` 个输出点；每 (id,ih) 平面只搬一次、被 windowWNum 个输出点共享。`windowWNum` 受 UB 预算与输出行边界（不跨 OW 行）约束。

**生成建议**：对窗口较小、输出点数多的 adaptive 场景，优先实现此「平面加载复用」；通用窗口逐点路径保留为兜底。

## 5. 三种 UB mode 路由（官方 tiling，可简化但要知道）

官方按 UB 压力三选一：
- `2*AlignUp(C) > tileLen` → **SPLIT_C**（C 过大，按 C 分片累加）
- `inputTileNum < maxWindowWLength` → **SPLIT_W**（窗口 W 塞不进输入 tile，按 W 分片）
- 否则 → **MULTI_W**（多输出点共享平面，§4）

生成算子用单模式（vectorLen C 分片 + 逐点窗口）就覆盖了全部 6 场景且 case[1-5] >1.1x——**简化是可行的**，但当 C 很大或窗口很大时需回到 SPLIT_C/SPLIT_W 思维。

## 6. 多 dtype 与 C 对齐

- 官方支持 fp16/bf16/fp32：非 fp32 先 `Cast` 到 fp32 累加，bf16 回写用 `CAST_RINT`（round-to-nearest），fp16 用 `CAST_NONE`。
- 生成仅 fp32 且硬性 `C%8==0` 守卫。对 ≥220 的 AI core，`DataCopyPad` 自动 pad 可放宽该守卫；非对齐 tail 在旧 core 上才需要 GatherMask/atomicAdd。
- 经验：**先用 C%8==0 守卫把正确性跑通，再放宽 tail**；守卫本身不是缺陷。

## 7. aclnn 中毒：自定义 kernel 与 aclnn 同进程共存的通用约束（最易复用的坑）

**症状**：`F.adaptive_avg_pool3d`（aclnn）之后立即用裸 `aclrtlaunch` 提交自定义 kernel → launch 被静默丢弃 + 进程持续中毒（后续所有 launch 全失败，输出未初始化/残留数据）。

**已排除**：后置 sync、前置 `aclrtSynchronizeStream`、前置 `aclrtSynchronizeDevice`、y=zeros vs empty。

**根因**：torch_npu aclnn 经内部 enqueue 机制延迟提交 kernel；aclnn 仍在飞时裸 aclrtlaunch 与之冲突。

**修复（必须）**：launch 前 `(void)c10_npu::npuSynchronizeDevice();`（torch_npu 层同步 = python `torch.npu.synchronize()`，排空 aclnn enqueue 队列）。**裸 ACL 层 sync 无效，必须 torch_npu 层。**

**代价**：每次调用 ~100us host 开销，短算子（<100us 桶）的 wall-time 性能被拖累到 ~0.3x。评测计时要注意区分 kernel time 与 wall time。

## 8. TileLang 层天花板：adaptive 窗口对 0.1.4 不友好，性能落 Phase 4

- 基线 0.1199x；核数扩展、T.Pipelined 均无收益；3D T.Parallel 被框架拒绝（`ascend_lower_parallel_to_vector.cc:299`）。
- 探针证实 ~355us 固定每调用开销 → Amdahl 上界 ~0.20x。**adaptive pooling 的 TileLang 性能迭代应尽早判框架限制并移交 Phase 4 手写 AscendC**，避免空耗 p_retry。
- 与 MaxPool3D 结论一致：池化类算子性能由 Phase 4 AscendC 承载，TileLang 只负责语义蓝图。

## 9. 正确性验证清单（adaptive 专属补充）

- [ ] 窗口索引整数公式对齐官方（MERE 目标 0，允许极小累加序差异）
- [ ] 覆盖 reduce_all / 常规 / 非均匀窗口 / **upscale（O>I）** / 大 C / 大 shape 6 类场景
- [ ] reduce_all 快路径必须过 multi-batch 随机输入（防大块搬入读错数据）
- [ ] aclnn 参考调用之后的自定义 kernel 必须验证不被静默丢弃（防中毒）
