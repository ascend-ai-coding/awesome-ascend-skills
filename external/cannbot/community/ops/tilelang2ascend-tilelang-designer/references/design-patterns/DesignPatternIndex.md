# 归约 / 重排 / 排序TopK / Norm 族 / 克隆搬运类算子设计模式索引（Design Pattern Index）

生成归约族、重排/搬运类、排序/TopK/采样类、Norm 族（含激活融合）或克隆/全量搬运主导类算子前先读本文件，用它决定采用哪套设计结构。模式可以组合，但不要临时发明新模式。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的模式，再只读对应文档的 `设计要点`；只有实现细节不够时，再继续读 translator references 中对应的实现指南。

## 归约族设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| 沿任意维（或全部）`sum / mean / max / min / prod` | `@references/design-patterns/references/reduce_design.md` — (O,R,I) 三分解路由选路径 |
| 均值/方差统计量（`layer_norm` / `rms_norm` / `batch_norm` / `var` / `std`） | 同上（forward 内含归约，落入同一套路由） |
| 需算方差/二阶矩（`var` / `std`，如 norm 类的归一化分母） | reduce_design — 二阶矩算法路线（同趟 sum+sumsq vs 两趟/Welford，按动态范围选） |
| 非末轴归约（内维 I > 1） | reduce_design — Path A 跨行 RA，沿 I 分块 iTile，不物理转置 |
| 末轴归约、单行可装 UB | reduce_design — Path B 多行批归约（AR） |
| 末轴归约、单行超 UB | reduce_design — Path C 分块两级树 |
| 行尾需对齐/补零 | reduce_design — tile 内 pad 语义规划（sum→0、max→-inf、min→+inf、prod→1） |

## 重排/搬运类设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| 奇偶交织 / 规律重排 / stride 切片重组（`chunk`/`split`/`cat`/`stack` 半区拆分） | `@references/design-patterns/references/shuffle_design.md` — 规律 pattern 结构（预留硬件指令映射），不建表 |
| gather / scatter / index 离散取数写 | shuffle_design — 通用结构，避免每-launch 建表 |
| 广播消费（cos/sin S=1、RoPE 交织、RotaryMul） | shuffle_design — 多行共享源行的 tile 数据流（源行去重） |
| 输出需按指定布局 | shuffle_design — 计算即按最终布局摆放，写回退化为连续搬移 |
| 小 shape（固定开销主导） | shuffle_design — 结构选择避开每-launch 固定开销 |

## 排序 / TopK / 采样类设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| `sort / topk / top_k_top_p / sampling` / 找第 k 大 / 阈值过滤 | `@references/design-patterns/references/sort_topk_design.md` — TopK 结构路由（**先过 §1.0 否决与估算**，再定值域二分 vs 排序路径） |
| 输出为**有序值 / 有序索引**（argsort 取 top-k 下标、gather 按序消费，如 MoE gating） | 排序路径（整行 sort 或分段 sort + 归并）——值域二分只产阈值/掩码，**一票否决** |
| 分组小 N（每组 ≤1024）求组内 top-k / top-k 和 | 排序路径——每组硬件 sort 近乎免费，二分多遍扫描纯亏；k≤2 的 reduce-max 替代也须先做硬件效率估算，禁止默认更快 |
| 输出为掩码/阈值过滤，且 N 较小 / 需全局统计量 / UB 装不下整行 | sort_topk_design — 值域二分（O(iter·N) 免排序），迭代下界按含 padding 的 range 算；**选路前必须完成 §1.0 复杂度+硬件效率估算** |
| 掩码/阈值过滤且 N 大（>≈8192） | sort_topk_design — 分段 sort + 归并取 top-k（单遍） |
| 参考实现用 `sort(ascending, stable)` + `mask < kth`（保留所有 tie） | sort_topk_design — tie 语义对齐（值阈值 vs 位置阈值，值域二分天然对齐；分段 sort 需重收集 `>= kth`） |
| top-k + top-p 组合过滤 / 采样（`p` 概率阈值） | sort_topk_design — p∈[0,1]、k≤min(N,1024) 输入约束；升/降序 cumsum 方向等价 |

## 决策顺序

多个模式同时出现时，按这个顺序确定设计：

```text
1. 归约：先定 (O,R,I) 三分解与路径（A 跨行 RA / B 多行批归约 / C 分块两级树）。
2. 重排/搬运：再定取数/广播/布局结构（pattern vs 建表、源行共享、最终布局）。
3. 排序/TopK：定 TopK 结构路由（值域二分 vs 分段 sort，按 N 阈值分野 + tie 语义）。
4. 克隆/全量搬运：定克隆段结构（独立 memcpy kernel 还是融合跳过；跨核写依赖 → 双 kernel）。
5. 共同：最后定 host 核数分档（按规模带，避免 dispatch ramp）。
```

> 实现层细节（GatherMask 指令、isReuseSource、dstStride、dtype 分路、rightPadding 参数等）
> 在转译阶段处理，见 translator references：`ascendc_reduce_patterns.md` / `ascendc_shuffle_patterns.md` / `ascendc_sort_topk_patterns.md`。

## Norm 族 + 激活融合设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| `group_norm` / `layer_norm` / `rms_norm` / `batch_norm` / `instance_norm`（含 +swish/silu/gelu 融合） | `@references/design-patterns/references/norm_fusion_design.md` — 组间独立归一化并行范式 + 组内两遍 |
| 归一化后接 per-channel affine（γ/β） | norm_fusion_design — affine 融合（预计算 `scale=γ·rstd`、`bias'=β−γ·mean·rstd`，Pass2 退化为一次乘加） |
| 两遍读同一输入（Pass1 归约 + Pass2 apply） | norm_fusion_design — 双缓冲软流水（输入/输出各 depth 2 重叠搬运与计算） |
| 二阶矩求 var/std，且输入大动态范围 | norm_fusion_design — 两遍中心化 + mean 修正（在 reduce_design §4b 之上细化落地结构） |
| 空间维 S 很大 / 需按 S 分片防 UB 越界 | norm_fusion_design — 按规模分档；实现层见 `ascendc_norm_fusion_patterns.md` §6 |

> Norm 族实现层细节（TQue 双缓冲、affine 预重排、swish 简洁形式、S_CHUNK 分片）在转译阶段处理，
> 见 translator references：`ascendc_norm_fusion_patterns.md`。

## 克隆 / 全量搬运主导类设计模式

| 如果看到 | 设计采用 |
| --- | --- |
| `output = clone(input)` + 少量更新（量化 scatter / index_put / 部分行覆写），克隆:更新流量比 ≥100:1 | `@references/design-patterns/references/shuffle_design.md` 决策 7 — 克隆段当独立 memcpy kernel 设计；跨核写依赖 → 双 kernel 拆分；实现层铁律见 `ascendc_quantization_patterns.md` §4 |
| 全量 fill / 全量 copy / 大张量 dtype 转换等纯搬运形态 | shuffle_design 决策 7 — memcpy 结构（全核铺满、大段、双缓冲），禁微段与 PIPE_ALL 风暴 |

> 克隆/搬运实现层铁律（大段 DataCopy(Pad)、TQue depth≥2、count 32B 对齐根因）在转译阶段处理，
> 见 translator references：`ascendc_quantization_patterns.md` §4 / `ascendc_shuffle_patterns.md` §3。
