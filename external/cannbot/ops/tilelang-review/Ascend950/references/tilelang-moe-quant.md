# TileLang MoE 与量化算子领域代码检视规则

<适用>
语言: Python, TileLang DSL
侧别: All
领域: true
触发: moe, expert, topk, top_k, stable_topk, quant, scale, sf_block, packed_ue8m0, FP4, FP8
默认启用: true

排除场景: 仅变量名或注释出现 expert/quant，实际没有 MoE 或量化逻辑
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 2
</检视负载>

## 目的

检查 TileLang MoE 路由与低精度量化相关问题。

## 快速索引

| 规则 | 标题 | 适用场景 | 严重级别 |
|------|------|---------|---------|
| MQ-01 | Expert Index 边界 | MoE | 高 |
| MQ-02 | MoE 参数和输出合约 | MoE | 高 |
| MQ-03 | 稳定 TopK、mask 和归一化语义 | MoE、TopK | 高 |
| MQ-04 | 量化 dtype 与逻辑/物理存储一致 | Quant | 高 |
| MQ-05 | Scale block、布局、stride 和 packed 协议一致 | Quant | 高 |
| MQ-06 | Scale 计算、rounding、饱和与尾块精度正确 | Quant | 高 |

## 范围概览

> 本文件覆盖 TileLang 中的 MoE 专家索引、参数合约和量化精度。
>
> **规则分类概览**：
>
> | 分类 | 规则范围 | 核心检视关注点 |
> |------|---------|-------------|
> | MoE 专家路由 | MQ-01~03 | 专家索引与映射、参数合约、稳定 TopK 和 mask 语义 |
> | 量化精度 | MQ-04~06 | dtype/存储协议、scale layout、scale 计算与精度保护 |
>
> **适用场景**：TileLang MoE 和量化算子的代码检视

## 术语表

| 术语 | 含义 |
|------|------|
| Expert Index | Token 选择的专家编号，合法范围 `[0, num_experts)` |
| Scale Layout | 量化 scale 的 block shape、packed 形式和行列布局 |
| Logical/Physical Shape | 公开数值元素 shape 与 packed storage 实际占用 shape |

## PR 差异→规则速查

> **使用方法**：查看 PR 差异代码中的关键词，匹配下表，仅阅读对应分类的规则。无需一次阅读全部规则。

| 差异关键词 | 对应分类 | 应读规则 | 典型搜索命令 |
|-----------|---------|---------|------------|
| `expert` / `topk` / `mask` / `routing` | MoE 专家路由 | MQ-01~03 | `rg -n -i 'expert|topk|routing|mask' <operator_path> <test_path> -g '*.py'` |
| `quant` / `scale` / `FP4` / `FP8` / `sf_block` | 量化精度 | MQ-04~06 | `rg -n -i 'quant|scale|sf_block|packed|fp4|fp8' <operator_path> <test_path> -g '*.py'` |

## 领域判定规则

### 核心特征（满足任一即可）

- 修改当前算子中 expert 选择、映射或 TopK。
- 修改当前算子中的量化格式或 scale layout。

### 排除场景

- 普通单卡 TopK，不涉及 expert index 或 MoE 参数。
- 普通 cast，不涉及量化格式、scale 或低精度数值语义。

---

## 一、MoE 专家路由规则

### MQ-01: Expert Index 边界 `[适用: All]` `[红线]`

**问题描述**

有效 expert index 必须满足对应逻辑或物理专家空间的 `[0, num_experts)`。Padding sentinel、负数、未初始化 index 和尚未映射的逻辑 expert id 不得参与物理地址计算。

**检查方法**

1. 追踪 index 的产生、tie-break、padding、TopK、逻辑到物理映射、写回和下游消费全链路。
2. 对每个阶段明确 index 所属空间、合法上界和 sentinel；不能用同一个 `num_experts` 模糊表示不同空间。
3. 检查 `1 <= num_topk <= num_experts`，以及 shared/duplicate expert 扩展后的输出宽度。
4. 动态 index 的范围检查必须发生在实际 GM load/store 前；先访问再 mask 无效。

**排除规则**

sentinel 只写入接口明确规定的无效输出位置，且所有 consumer 在地址计算前可靠过滤时可以接受。

**判定方法**

非法 index 可到达有效输出或地址计算、映射表索引越界、或逻辑/物理专家空间混用时判 FAIL。

---

### MQ-02: MoE 参数和输出合约 `[适用: Host]` `[红线]`

**问题描述**

MoE wrapper、Kernel factory、Kernel signature、reference 和可选 `out` 参数必须对 expert 数、topk、score dtype、映射表及输出 shape/dtype/device 使用同一合约。只在 Kernel 内 assert 会让公开入口在编译或运行较晚阶段失败。

**检查方法**

- 检查 logits/score 的 rank、dtype、device、contiguous/stride 和 expert 轴位置。
- 检查 `num_topk`、shared expert、group/expert 数的下界、上界、整除和互斥组合。
- 成对核对 bias/mask、mapping/count、fix-routing/unmapped-index 等可选参数；一侧存在时另一侧是否必需。
- 核对新分配输出和用户传入 `out` 的 shape、dtype、layout、device 与空 token 返回路径。

**判定方法**

公开允许输入在 wrapper、Kernel 或 reference 任一层含义不一致，或输出 buffer 合约不一致时判 FAIL；完整入口校验能够支配调用时 PASS。

---

### MQ-03: 稳定 TopK、mask 和归一化语义 `[适用: All]` `[红线]`

**问题描述**

TopK 不只是选择最大值：相等分数的稳定次序、NaN/Inf、bias、mask 优先级、fixed/random routing、归一化与 scaling 顺序都属于公开数值语义。改变比较符号或 padding 值可能只在重复值、全零行或尾专家组上失败。

**检查方法**

1. 从公开 wrapper/reference 确定 tie-break；本仓库 `topk_gate` 的相等分数语义是选择较小 expert index。
2. 核对 padding lane 使用不会胜出的值，并在 index/weight 写回前清理无效 lane。
3. 对 mask、force-random、fixed-routing、bias/image-bias 建立优先级表，检查所有组合和空 token 路径。
4. 核对 softmax/sigmoid/identity 等 scoring 与权重归一化、`routed_scaling_factor` 的先后顺序。

**排除规则**

若某特殊值、随机路径或归一化模式不属于公开接口，入口必须明确拒绝或路由到支持实现；不能仅因测试未覆盖而默认排除。

**判定方法**

稳定顺序、mask/sentinel、scoring 或归一化与公开 reference 不一致时判 FAIL；随机分布质量需要可复现统计测试，不能只凭代码形态下结论。

---

## 二、量化精度规则

### MQ-04: 量化 dtype 与逻辑/物理存储一致 `[适用: All]` `[红线]`

**问题描述**

量化格式名、torch dtype、TileLang dtype 和实际 storage dtype 不一定一一对应。例如 packed FP4 可由较宽整数 storage 承载，逻辑 hidden 与物理元素数不同；把 storage dtype 当作数值 dtype 会造成 shape、索引和输出协议错误。

**检查方法**

- 从配置对象追踪输入/输出格式到 torch dtype、TileLang dtype、storage dtype 和 pack factor。
- 核对 `T.view`/torch view 前后总字节数、对齐和 logical/physical hidden 换算。
- 检查 wrapper、Kernel、reference、consumer 和输出类型标注是否使用同一格式语义。

**判定方法**

无依据改变 dtype、pack factor 或 logical/physical shape，导致有效元素映射改变时判 FAIL。`T.reinterpret` 只允许明确位协议，不代替数值转换。

---

### MQ-05: Scale block、布局、stride 和 packed 协议一致 `[适用: All]` `[红线]`

**问题描述**

scale Tensor 的 `sf_block`、shape、row/column-major、stride、packed/unpacked 和转置后视图必须在生产与消费两端一致。设备相关 pack factor 应由当前仓库 helper/配置获得，不能从其他后端照搬。

**检查方法**

1. 用输入 logical shape 和 `sf_block` 独立计算预期 scale shape。
2. 追踪 col-major 转置、packed UE8M0 view、切片和 epilogue 后用户看到的最终 layout。
3. 核对动态 stride 的单位与连续维，尤其是短 scale 行和尾 block。
4. 检查所有支持配置的 wrapper dispatch 与 Kernel specialization 一致。

```python
expected_sf_shape = get_sf_shape(input_shape, config)
assert scale.shape == expected_sf_shape
```

**判定方法**

生产和消费端对同一 scale 元素位置、pack word 或 stride 的解释不一致时判 FAIL；仅凭 Tensor 总元素数相同不能排除布局错误。

---

### MQ-06: Scale 计算、rounding、饱和与尾块精度正确 `[适用: Kernel]` `[红线]`

**问题描述**

amax/scale 的统计 dtype、零 amax 下限、power-of-two rounding、饱和范围和量化/反量化顺序共同决定误差。尾块的无效 lane 若参与 amax 会污染整个 block 的 scale；packed 输出未清理则可能泄漏未初始化 bit。

**检查方法**

- amax 和 scale 中间计算使用满足 reference 的精度，并对全零 block 使用配置规定的正下限。
- 核对 round-to-power-of-two、倒数 scale、格式最大值和最终 cast 的先后顺序。
- 检查 FP4/FP8 溢出、下溢、NaN/Inf、正负零与随机 rounding 语义。
- 尾块统计只包含有效输入；无效 packed lane 在写回前按接口协议清理。

**证据要求**

使用覆盖零块、极值、正负值、非整 block、packed/unpacked 和往返反量化的 fp32 reference 测试。不能通过放宽容差或屏蔽有效 lane 获得通过。

**判定方法**

实现公式、rounding、饱和或有效 lane 结果与公开 reference 不一致时判 FAIL；只有性能差异而精度正确时不在本条例中报告。

## 搜索关键词总表

```bash
rg -n -i 'expert|topk|routing|quant|scale|sf_block|packed|fp4|fp8' <operator_path> <test_path> -g '*.py'
```
