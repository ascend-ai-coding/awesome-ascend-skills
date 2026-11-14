# 技术卡片库索引

> 每张卡片结构一致：适用 bound → 何时用(诊断) → 原理 → 怎么改(before/after) → 性能数据 → 红线 → 关联。
> 采集与指标口径见 `../metrics.md`（统一 msprof）。诊断入口见 `../index.md`。
> 通用方法由 arch22 / arch35 共用；只按表中标注选择代际相关实现。RegBase / Reg / MicroAPI / VF 为 arch35 独占，代码示例与实测收益不自动跨代生效。
>
> **卡片编号（大类-小类）**：文件名 `<大类>-<小类序号>-<短名>.md`。大类按 bound 类型：`vec` / `scalar` / `mem` / `nobound`（与四个诊断页对应）；小类序号在各大类内**独立、零填充、只增不改**。新增卡片直接在对应大类追加下一个序号（如新 VEC 卡 = `vec-15-*`），**不跨大类改号、不回填**；交叉引用用卡片 ID（如 `vec-05`）。
>

| 卡片 | 适用代际 | 优化点 | 适用 bound | 来源 | 状态 |
| --- | --- | --- | --- | --- | --- |
| [vec-01-ub-fusion](vec-01-ub-fusion.md) | 通用方法 | UB 内融合，消除 GM 往返 | VEC / 访存 | vec.md 策略1 | 写满 |
| [vec-02-reduce-cast](vec-02-reduce-cast.md) | 通用；VF 方案仅 arch35 | 减少/批量 Cast | VEC | vec.md 策略2 | 写满 |
| [vec-03-fused-instr](vec-03-fused-instr.md) | arch35（VF） | 融合指令 MulDstAdd/VMULA | VEC | vec.md 策略3 + VF 策略2 | 写满 |
| [vec-04-scalar-roundtrip-vectorize](vec-04-scalar-roundtrip-vectorize.md) | 通用方法 | GetValue/SetValue→Brcb+批量算 | VEC / Scalar | VF 策略3 | 写满 |
| [vec-05-eliminate-ub-staging](vec-05-eliminate-ub-staging.md) | arch35（VF） | 删中转 UB，寄存器内直算 | VEC / 访存 | VF 策略4 | 写满 |
| [vec-06-broadcast-once-vl-merge](vec-06-broadcast-once-vl-merge.md) | arch35（VF） | 不变量广播一次+多行 VL merge | VEC / 访存 | VF 策略5 | 写满 |
| [vec-07-low-latency-reduce](vec-07-low-latency-reduce.md) | 通用方法 | 硬件树形归约 | VEC | vec.md 策略4 | 写满 |
| [vec-08-multi-accumulator-unroll](vec-08-multi-accumulator-unroll.md) | arch35（VF） | 多路累加破归约依赖链 | VEC / 无 bound | VF 策略8 | 写满 |
| [vec-09-loop-sink-into-vf](vec-09-loop-sink-into-vf.md) | arch35（VF） | 外层 for 下沉进 VF | VEC / Scalar | VF 策略9 | 写满 |
| [vec-10-small-loop-unroll](vec-10-small-loop-unroll.md) | arch35（VF） | 小定长循环全展开 | VEC / 无 bound | VF 策略7 | 写满 |
| [vec-11-reduce-compute-lut](vec-11-reduce-compute-lut.md) | 通用方法 | LUT+插值减计算量 | VEC | vec.md 策略6 | 写满 |
| [vec-12-preg-split](vec-12-preg-split.md) | arch35（VF） | preg 溢出按条件拆 VF | VEC / 无 bound | VF 策略10 | 写满 |
| [vec-13-vf-tail-unified-mask](vec-13-vf-tail-unified-mask.md) | arch35（VF） | 尾块统一 mask 折进末拍，去独立尾块循环 | VEC / Scalar | RoPE arch35 | 写满 |
| [vec-14-aligned-split-copy](vec-14-aligned-split-copy.md) | arch35（VF） | 拆分搬运让各段 UB 对齐，VF 走对齐微指令 | VEC / 访存 | RoPE arch35 | 写满 |
| [scalar-01-scalar-bound](scalar-01-scalar-bound.md) | 通用方法 | Scalar bound 手段清单 | Scalar | scalar.md | 占位 |
| [mem-01-aligned-contiguous-copy](mem-01-aligned-contiguous-copy.md) | 通用方法 | 对齐并连续化主路径，隔离非对齐/尾块 | 访存 / MTE | #1，5 次：ops-nn + ops-transformer | 写满 |
| [mem-02-ping-pong-prefetch](mem-02-ping-pong-prefetch.md) | 通用方法 | 双 buffer 预取下一 tile，隐藏搬运 | 访存 / 流水 | #2，5 次：ops-nn + ops-transformer | 写满 |
| [mem-03-batched-row-copy](mem-03-batched-row-copy.md) | 通用方法 | 按 UB 预算读一次多行、合并同生命周期队列 | 访存 / MTE | #3，4 次：ops-nn + ops-transformer | 写满 |
| [mem-04-ub-buffer-reuse](mem-04-ub-buffer-reuse.md) | 通用方法 | UB 内复用临时缓冲，删 GM 清零/中转 | 访存 / UB | #4，4 次：ops-transformer | 写满 |
| [mem-05-l2-bypass-one-shot](mem-05-l2-bypass-one-shot.md) | 通用方法 | 仅对可证明一次性的数据绕过 L2 | 访存 / Cache | #5，3 次：ops-nn + ops-transformer | 写满 |
| [mem-06-direct-contiguous-output](mem-06-direct-contiguous-output.md) | 通用方法 | 连续完整结果一次性直写，删冗余搬出 | 访存 / MTE | #6，3 次：ops-transformer + ops-math | 写满 |
| [mem-07-identity-pure-copy](mem-07-identity-pure-copy.md) | 通用方法 | identity/退化语义转 pure-copy 快路径 | 访存 / Scalar | #7，3 次：ops-math | 写满 |
| [nobound-01-nobound](nobound-01-nobound.md) | 通用方法 | 无 bound 手段清单 | 无 bound | no-bound.md | 占位 |
| [nobound-02-stage-order-rescheduling](nobound-02-stage-order-rescheduling.md) | 通用；仿真示例仅 arch35 | 三阶段顺序重排：计算发射优先 + 消费阶段延后一拍 | 无 bound | RoPE/KVCache 融合算子 arch35 | 写满 |
