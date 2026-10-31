# {operator_name} apace（MC2 通算融合）算子设计文档

本模板是 `/ascendc-mc2-best-practice` apace 路线的唯一 DESIGN 模板，由 Step 3 使用，生成：

```text
<project-root>/operators/<operator_name>/docs/DESIGN.md
```

- 路线判定为 `apace_native` / `apace_custom`：完成全部章节并通过 §6 确认清单后，按 `plan-template.md` 生成 PLAN。
- 路线判定为 `unsupported`：只保留 §0 需求与事实、§5 不支持项及阻塞证据，不生成 PLAN。
- §0.3 核对事实不足以支撑设计：在相应条目标记 `blocking` 并停止，等待补充核对或用户澄清，不产出 DESIGN/PLAN。

> 生成文档时替换全部占位内容；只写入当前用户需求与已核对的 apace 源码事实，模板示例、官网样例、旧项目或候选算子名称均不能作为能力、接口或验证证据。本模板只约束新生成的 DESIGN/PLAN，既有算子的历史设计文档不要求回溯改写。

---

## 0. 概述

### 0.0 需求判定

| 判断项 | 结论 | 依据 |
|-------|------|------|
| 需求类型 | 特定用例 / 通用 | 明确给出 shape、dtype、rankSize 为特定用例；否则按通用处理 |
| 用户目的 | 算子设计 / 完整算子开发 | 决定执行到设计步骤还是完整四步；不影响本文档内容 |
| 需求是否明确 | 是 / 否 | 不明确时先向用户澄清；澄清前不做路线判定 |

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | `{operator_name}` |
| 算子类别 | `<如 all_gather_quant_matmul / quant_matmul_reduce_scatter / all_to_all_quant_matmul / 新增>` |
| 通信方向 | GET / PUT / compute-first |
| 集合通信原语 | AllGather / AllToAll / ReduceScatter / `<其他>` |
| 需求类型 | 特定用例（shape=`<...>`，dtype=`<...>`，rankSize=`<n>`） / 通用 |
| 支持数据类型 | `<输入、scale、累加、中间值和输出 dtype，如 fp8_e4m3fn + e8m0 scale → bf16>` |
| 支持 shape/layout | `<逻辑 shape、物理 layout、transpose 和动态轴>` |
| rankSize 与组网 | `<rank 数量合法域、卡间拓扑、UDMA 建链约束>` |
| 目标芯片与 NpuArch | `<target_chip>`，`<npu_arch>`（编译参数 `--npu-arch=<npu_arch>`） |
| CANN 版本 | `<cann_version 或未指定>` |
| 特殊约束 | `<精度、性能、Win 区容量、接口或部署约束>` |
| 待澄清项 | `<无，或列出 blocking 项>` |

### 0.2 用户原始需求

按用户原话逐条记录需求，不改写、不拆分、不合并；编号确定后不再变动，后续设计核对、PLAN 用例追溯与验收均引用本节编号。

| # | 需求内容 |
|---|---------|
| 1 | `<用户原文>` |
| 2 | `<用户原文；无则删除本行>` |

### 0.3 调查事实（设计前核对结果）

事实来源 = CANN 内置 apace 框架只读核对（清单见 [`workflow/step2-investigation.md`](../workflow/step2-investigation.md)），**默认内联记录于本节**；每条事实带 `文件:行号` 引用：

| 事实类别 | 内容 |
|---------|------|
| matmul 链路事实 | `<Blaze 组件、dtype 组合、M/N/K、localMatmul 候选、tiling 基线；逐条带 文件:行号>` |
| 通信接口事实 | `<CollectiveComm 四段式契约、GET/PUT 钩子、CommContext 字段、Win 区布局>` |
| 入口 ABI 事实 | `<入口签名、CommContext 传递方式、dtype 变体入口数>` |
| 官方覆盖性 | `<官方 kernel 可直接调用/复用的部分；未覆盖的需求项（gap 必须有不兼容/拒绝证据或穷尽读取边界后的无匹配事实）>` |
| 未闭合项 | `<none，或列出 blocking 项>` |

### 0.4 约束显式确认（门禁必查）

| # | 约束 | 结论 | 证据 |
|---|------|------|------|
| ① | 禁止 `__schedmode__(1)` | confirmed / blocking | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| ② | Matmul 走 Blaze 模板 | confirmed / blocking | 通信在前：vendor `BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`；compute-first：自研 FragmentTensor kernel；均无 `AscendC::Matmul` |
| ③ | 禁止修改 `block/` 和 `tiling/` | confirmed / blocking | 只在 `operators/<operator_name>/` 下创建文件，共享层 CMake 直引零复制 |
| ④ | 直调仅 UDMA | confirmed / blocking | HCCL windows 不支持直调 |

---

## 1. 数学定义与数据流

### 1.1 数学语义（每 rank 视角）

```text
逻辑输入:
  <输入名>: shape=<...>, dtype=<...>, layout=<...>, 角色=<数据 | scale | 权重 | 偏置 | ...>
逻辑输出:
  <输出名>: shape=<...>, dtype=<...>, layout=<...>, 角色=<...>

每 rank 数学语义:
  <逐步公式；写清本地分片在切分轴上的范围、跨 rank 聚合方式（拼接/求和）、归约顺序与类型转换顺序>
```

| 步骤 | 数学/通信操作 | 输入 dtype | 累加/计算 dtype | 输出 dtype | 对应需求 # |
|-----|--------------|------------|----------------|------------|-----------|
| `<n>` | `<操作>` | `<dtype>` | `<dtype>` | `<dtype>` | `<ref>` |

### 1.2 golden 语义（每卡输入/输出契约，门禁必查）

逐 rank 明确输入分布、输出语义与切分轴——这是 gen_data.py golden 的依据，切分轴写错则精度验证整体失效。**输入分布轴与输出分布轴必须分别冻结**（二者可不同：如输入不切分、输出沿 M 分布）；未明确时回 grill-protocol 维度 9 澄清，禁止默认：

| rank | 本地输入（data/scale 分片） | 输入分布轴（输入按哪轴切/不切） | 远端数据来源 | 输出语义 | 输出分布轴（输出按哪轴分布） | 聚合方式 |
|------|---------------------------|------------------------------|-------------|---------|------------------------------|---------|
| `<i>` | `<本地分片公式或"完整输入">` | `<M/N/K/不切>` | `<来自哪些 rank 的哪些分片>` | `<输出张量的哪一段>` | `<M/N/K/不切>` | `<拼接/求和/无>` |

```text
golden_formula_per_rank: <从逻辑输入计算的 CPU golden 公式，含累加 dtype 与舍入顺序>
quantization_decode_rule: <若设备输入为 MX/FP8 编码字节，须从最终写入设备的实际字节解码后计算 golden>
```

### 1.3 数据流

AIV 侧（通信）与 AIC 侧（计算）的完整数据流向：GM → Win 区 → 消费 → 输出；标注每段的生产者/消费者与同步点：

```text
<AIV 数据流：源 GM → Commit → 远端 Win → Wait → 本 rank 消费点>
<AIC 数据流：Win 区/GM → L1/L0 → L0C → fixpipe → staging/输出 GM>
同步点：<CrossCore flag 的生产者→消费者配对位置>
```

---

## 2. 技术路线决策

### 2.1 capability-declaration 查询

| 决策栈 | 命中行 |
|-------|-------|
| chip × 算子类型 × 调用形态 × 通信路径 × 编程抽象 | `<supported 行编号与内容；命中否定行或无行 → 本 DESIGN 判 unsupported>` |

### 2.2 官方 kernel 覆盖性

使用 §1 需求逐条对照官方 kernel/block 的覆盖情况。未覆盖项的成立条件：官方接口存在明确的不兼容/拒绝证据，或在已声明的读取范围内逐文件核对后确认无匹配实现；仅在个别文件中没找到不构成未覆盖：

| 需求 # | 精确需求 | 官方 kernel/block 覆盖情况 | 结论 | 证据（文件:行号） |
|-------|---------|---------------------------|------|------------------|
| `<n>` | `<需求>` | `<已验证组件>` | covered / gap | `<ref>` |

### 2.3 路线决策

| 条件 | 结论 | 证据 |
|------|------|------|
| 官方 `apace/kernel` 可直接调用/参考复用 | apace_native / 否 | `<refs>` |
| 官方 kernel 未覆盖 + `apace/block` 接口可组合 | apace_custom / 否 | `<refs>` |
| 场景注册表命中 | 语义命中 `<scenario>` / 零命中 / 多命中 / not_applicable | `<refs>` |
| **最终路线** | **apace_native / apace_custom / unsupported** | `<refs>` |

- `apace_native`：不读取场景注册表，直接参考官方 kernel 实现组织工程。
- `apace_custom`：场景语义命中，默认查阅场景指导，记录 `selected_scenario` 与场景指导来源；授权修改范围仅限 `operators/<operator_name>/` 下新建文件。
- `unsupported`：只生成本阻塞 DESIGN，逐项写入 §5.3。

### 2.4 API 验证清单（门禁必查）

每个被选用的 API/组件必须完成验证；未验证 API 禁止入设计，未确认项返回 Step 2 补充核对：

| API/组件 | 验证来源（官方文件:行号） | 当前 CANN 版本可用性 | 布局/功能/限制确认 | 匹配结论 |
|----------|--------------------------|---------------------|--------------------|---------|
| `<如 CollectiveComm<AllGather, PUT, ...>>` | `<文件:行号>` | confirmed / blocking | `<连续性、地址公式、barrier 语义>` | confirmed / blocking |
| `<如 BlockMmad / MatmulWithScaleMx>` | `<文件:行号>` | confirmed / blocking | `<模板参数、specialization>` | confirmed / blocking |
| `<如 TeamBarrier / CrossCoreSetFlag>` | `<文件:行号>` | confirmed / blocking | `<flagId、PIPE、totalJobs>` | confirmed / blocking |

---

## 3. 架构设计

### 3.1 AIV/AIC 分工与通信编排

| 项目 | AIV（通信） | AIC（计算） | 证据引用 |
|-----|------------|------------|---------|
| 职责 | `<Commit/Wait/Finalize、TeamBarrier>` | `<RunMatmul/Process、消费 Win 区 tile>` | `<refs>` |
| 编排形式 | 严格分离（前/后 R 核分工） / 时分复用（同核错位流水） | 同左 | `<refs>` |
| localMatmul 模式 | 通信在前：`<0/1/2 分支选择及依据>`；compute-first：N/A（无此字段） | — | `<refs>` |

通信合同：

| 项目 | 设计结论 | 证据引用 |
|-----|---------|---------|
| 通信方向 | GET / PUT / compute-first | `<refs>` |
| 集合通信原语与实现类 | `<如 CollectiveComm<AllToAll, PUT, AType, TeamBarrier>>` | `<refs>` |
| 通信轮次 T 推导 | kernel 侧 `commTurn = splitAxisTileCnt + splitAxisTailCnt`；host 侧派生规则（compute-first 默认 `T \| mSeg` 无尾块；单尾块 ≤ 头块 16 对齐可直传） | `<refs>` |
| rankSize 与组网 | `<rank 数、卡间拓扑、targetRank 映射（每通信核负责 1 个 targetRank 并行 PUT/GET）>` | `<refs>` |
| self rank 跳过规则 | `<DoCommit/DoWait 对 targetRankId==rankId 的处理>` | `<refs>` |
| Win 区预算 | 通信在前（多对象）：data 段 `rankSize × rankDataBytes` + scale 段 `rankSize × scaleKaSize × axisM`；compute-first（单通信对象）：`M × N × sizeof(CType)`；数据区偏移按 host 建链布局确定、host/kernel 同源 | `<refs>` |

Flag 编排（CrossCore flag 配对，门禁必查）：

| flagId | PIPE | 生产者（Set） | 消费者（Wait） | 语义 | 证据引用 |
|--------|------|--------------|---------------|------|---------|
| `<如 0x2>` | `<如 MTE3/MTE2>` | `<AIV 通信完成后 Set>` | `<AIC Wait 后消费 tile>` | `<配对语义、waitedMask 去重、尾部兜底>` | `<refs>` |

其余同步：

| 同步机制 | 使用点 | totalJobs/BarrierMode | 语义 | 证据引用 |
|---------|-------|----------------------|------|---------|
| `SyncAll<true>` | `<位置>` | — | `<语义>` | `<refs>` |
| `TeamBarrier` | `<位置>` | `<totalJobs>` | `<语义>` | `<refs>` |

### 3.2 切分与 Tiling 策略

| 项目 | 设计结论 | 证据引用 |
|-----|---------|---------|
| 切分轴（按 rank） | M / N / K / 不切；`<每 rank 本地分片公式>` | `<refs>` |
| `splitAxisTileCnt` 策略 | 两阶段：精度调试 `tileCnt=1` 串行基线；性能调优扫描 `{1,2,4,8,16,32}`（通信在前默认值；compute-first 计数式固定 flagId 下 T 上限按 case 实测确定，R≤32 为 FragmentTensor 容量约束） | `<refs>` |
| 三层 tiling 结构 | `<算子级 tiling_data + CommTilingData + Blaze matmul tiling 的字段与 host 填充>` | `<refs>` |
| tail 与空任务 | `<splitAxisTailCnt、尾部 tile、空分片行为>` | `<refs>` |

### 3.3 UB 与资源预算

```text
ub_budget: <静态分配表：commBuf + barrierBuf + 归约/计算区等，逐项字节数>
win_area_budget: <数据段总字节数，与 host 注册容量一致性>
grid_usedcore: <usedCoreNum、AIV/AIC 配比 KERNEL_TYPE_MIX_AIC_1_1、通信核数 = rankSize>
```

### 3.4 入口签名与 dtype 变体

```text
entry_symbols: <__global__ 入口符号与实例化变体；dtype 合同全部组合有对应入口 + host 运行期 dispatch>
ordered_parameters: <GM 参数顺序：hcommCtx/aGm/scaleAGm/bGm/scaleBGm/cGm/workspace/tilingData 等>
tilingdata_fields: <字段到 device consumer 的映射>
workspace_and_lifecycle: <owner、生命周期、复用条件>
```

### 3.5 逻辑到物理 buffer 转换合同

逻辑张量进入设备前发生物理形态变化的（按 rank 分片写入 Win 区段、对齐 padding、FP8/MX 量化编码字节等），逐参数记录一行；未发生转换的参数，其逻辑字节序列即设备输入事实源，不额外构造中间表示：

| logical_id | 转换 | physical_shape | byte_span | offset/stride | 首个设备消费者 | 同步 owner |
|-----------|------|---------------|-----------|---------------|--------------|-----------|
| `<如 aGm>` | `<逻辑 A → 各 rank 分片 → Win data 段>` | `<分片后 shape>` | `<字节数>` | `<winOffset/行 stride>` | `<AIC BlockMmad / AIV PUT 钩子>` | `<flag id>` |

任一转换后的物理 shape、字节数或偏移在 host 与 kernel 两侧口径不一致时，必须在 host 侧启动前阻断，不能等到设备精度失败再反推布局错误。Golden 必须按最终实际写入设备的物理字节解码后计算。

### 3.6 工程组织

| 项目 | 设计结论 |
|-----|---------|
| 文件布局 | `<operators/<operator_name>/ 下 kernel/ src/ scripts/（development-guide §1.2）>` |
| 构建方式 | CMake 直引 CANN 内置 apace（APACE_ROOT），共享层零复制 |
| 修改范围 | 仅 `operators/<operator_name>/` 内新建/修改；`apace_source_root` 共享层整体只读 |

---

## 4. 验证合同

### 4.1 精度标准

| 项目 | 标准 | 依据 |
|-----|------|------|
| 比对阈值 | `<rtol/atol 或 exact 规则>` | `<dtype 对应的精度标准>` |
| nonfinite 门 | `<NaN/Inf 处理策略>` | `<refs>` |
| 边界矩阵 | `<tail、极小/极大 shape、tileCnt 各取值的覆盖组合>` | `<refs>` |

### 4.2 多卡测试矩阵

| 维度 | 取值 | 说明 |
|-----|------|------|
| rankSize | `<如 2/4/8>` | `<覆盖合法域端点>` |
| shape 组合 | `<M/N/K 组合>` | `<对齐/非对齐/tail>` |
| tileCnt | `{1, 2, 4, ...}` | 先 `tileCnt=1` 串行基线，再扫描 |
| dtype 组合 | `<输入/scale/输出组合>` | `<refs>` |

### 4.3 性能基线

| 项目 | 设计结论 | 证据引用 |
|-----|---------|---------|
| 基线对照 | `<串行基线 / 官方样例 / 理论带宽>` | `<refs>` |
| L2 flush 证据 | `<性能采集前后的 L2 flush 手段与记录位置；无 flush 数据不得作为性能结论>` | `<refs>` |
| 采集方式 | `<msprof / 打点计时，须与精度验证分离>` | `<refs>` |

Step 3 不实现、不运行设备，也不得写设备 PASS。

---

## 5. 支持边界与红线

### 5.1 支持域

| 维度 | 支持域 | 拒绝域 | 拒绝语义 |
|-----|-------|-------|---------|
| dtype | `<支持组合>` | `<拒绝组合>` | `<host 校验报错>` |
| shape/对齐 | `<支持域>` | `<拒绝域>` | `<报错>` |
| rankSize/组网 | `<合法域与拓扑>` | `<非法域>` | `<报错>` |
| 通信方向 | `<本算子支持的方向>` | `<未支持方向>` | `<unsupported>` |

### 5.2 红线确认

| 红线 | 本算子适用性 | 确认 |
|------|-------------|------|
| 全局红线 + 场景约束（[`review-checklist.md`](../review-checklist.md)） | `<逐项标注 适用/N.A.；场景约束按 selected_scenario 选取>` | confirmed / blocking |

### 5.3 不支持项（仅 `unsupported` 路线必填）

| 需求 # | 未覆盖原因 | 场景匹配结果 | 证据引用 | 用户恢复路径 |
|-------|-----------|-------------|---------|-------------|
| `<n>` | `<gap 证据>` | 零命中 / 多命中 | `<ref>` | `<需求变更或待补证据>` |

---

## 6. 确认清单

- [ ] §0.2 原样、逐条记录用户需求，编号稳定。
- [ ] §0.3 调查事实逐条带 `文件:行号`，无虚构接口/文件名，未闭合项已标 `blocking`。
- [ ] §0.4 四项约束全部 confirmed，无 blocking 进入可执行路线。
- [ ] §1.2 golden 语义逐 rank 给出输入/输出分布与切分轴，与 gen_data.py 依据一致。
- [ ] 通信方向（GET/PUT/compute-first）、编排形式（严格分离/时分复用）、切分轴（M/N/K）均已冻结且有源码证据。
- [ ] 通信轮次 T 由 `splitAxisTileCnt + splitAxisTailCnt`（kernel）与 host 派生规则共同冻结，取值依据可解释。
- [ ] Win 区预算与布局（通信在前两段式 / compute-first 单对象）字节数与 host 注册容量一致，数据区偏移 host/kernel 同源。
- [ ] CrossCore flag 配对表完整：flagId、PIPE、生产者、消费者、去重/兜底语义。
- [ ] AIV/AIC 分工与同步点明确，核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证。
- [ ] §2.4 每个 API 标注验证来源（官方文件:行号）与可用性结论。
- [ ] §3.4 入口变体覆盖 dtype 合同全部组合，host 有运行期 dispatch。
- [ ] §4.2 多卡矩阵覆盖 rankSize/shape/tileCnt/dtype 端点。
- [ ] §4.3 性能基线含 L2 flush 证据要求。
- [ ] §5.2 全局红线与场景约束逐项确认完成（场景约束按 selected_scenario 选取）。
- [ ] `unsupported` 路线无 PLAN 绑定，§5.3 逐项记录不支持原因。
- [ ] 文档无未证实签名、固定 recipe、实现结果或设备 PASS。
