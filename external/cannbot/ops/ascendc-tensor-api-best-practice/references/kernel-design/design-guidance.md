# DESIGN 模板填写指导（每栏填什么、怎么填）

本参考在 Kernel Design 节点按 implementation_route=tensor_api_custom 条件加载，与 [design-template.md](design-template.md) 配套：模板给空表，本文给每栏的**填写内容、填写示例与去向**。

> 填写范本：`07_tensor_api/{matmul_bias_fusion, custom_op_tensor_api}`。本指导只讲「填什么」，各栏的技术细节交叉引用对应 reference，不重复展开。

## 0. 填写顺序（按依赖，不要从第 1 栏一路顺填）

1. 先填 **§1 用户合同**（数学定义 + 支持域 + 硬约束）——它决定后面所有栏。
2. 再填 **§3 数据流与 Layout**（tensor 表）——它是 §4 内存预算、§6 ABI、§7 同步的输入。
3. 由 §3 派生 **§4 内存预算** → **§5 Tiling 尾块** → **§7 同步生命周期**（三者互相约束，可迭代）。
4. 最后填 **§6 ABI Crosswalk**（把 §3/§4 落到物理 bytes）与 **§8 Golden 合同**（依赖 §6）。
5. §2 API 分工、§9 构建、§10 验证、§11 阻塞 可随时填，最后统一核对。

## 1. 逐栏填写指导

### §1 用户合同
- **数学定义**：写清楚算子公式（如 `C = A@B + bias`），非「算一个 matmul」这种模糊话。
- **逻辑输入/输出**：列每个张量的逻辑含义与 shape（如 `hidden_states [B,S,H]`），不是物理分形 shape。
- **dtype/shape/layout 支持域**：明确支持哪些 dtype、shape 范围（含 M/N/K 奇偶约束，如 split_m 要求 `base_m % 2 == 0`）、layout（ND/NZ/…）。这是「支持域」不是「本次实现」，两者分开写。
- **目标芯片 / NPU Arch**：`DAV_3510`；编译参数才用 `dav-3510`（见 [build-and-toolchain.md](../build-and-toolchain.md)）。
- **用户硬约束**：只写用户明确提的（单 Kernel、数据驻留、精度 1e-3、性能目标），不脑补。

### §2 API 分工
- 只分两行：Tensor/Layout/Copy/Mmad 归 Tensor API；初始化/同步/锁/事件/原子归 C API。
- 「证据」栏填头文件/手册/样例的具体出处，不写「官方说」。分工边界见 [api-dataflow-layout.md](../api-dataflow-layout.md) §API 边界。

### §3 数据流与 Layout（核心栏）
- 每行一个 tensor（含中间 buffer），**逻辑 shape 与物理 shape 分开**，Pattern 填 `nd_ext_layout_ptn`/`nz_layout_ptn`/`zn_layout_ptn`。
- **writer / reader**：谁写、谁读，唯一 writer 是硬门禁（[review-checklist.md](../review-checklist.md) 代码检视第 2 条）。
- **同步 owner**：谁负责该 buffer 的同步（一个 buffer 一个 owner），填了才能衔接 §7。
- 布局/Pattern/format 的取值见 [tensor-api-reference.md](../tensor-api-reference.md) 与 [api-dataflow-layout.md](../api-dataflow-layout.md)。

### §4 内存预算
- 按 Storage 分级（UB/L1/L0A/L0B/L0C/Bias）各算占用量，**必须过容量门禁**（L1 512KB、L0C 256KB、UB 256KB 单 Vector 基线，见 [performance-patterns.md](../performance-patterns.md) §4）。
- 每行算 `Elements × dtype bytes`，累加得 Total ≤ Capacity × 安全利用率。超限失败模式是「全挂 + 超时」，必须构建期 CMake 检查硬失败。

### §5 Tiling 与尾块
- 每轴填 tile、tail、alignment、policy；**tail 必须显式**（`(axis % tile) ? axis % tile : tile`），padding 不参与 golden。
- 主块长度反推、四场景选型见 [tiling-strategy.md](../tiling-strategy.md)；访存计算比再平衡见 [performance-patterns.md](../performance-patterns.md) §2.1。

### §6 ABI Crosswalk
- 每行填「逻辑对象 → 物理表示 → byte span → offset/stride → host 接线 → device 消费者 → 证据」，把逻辑 tensor 落到实际 H2D 物理 bytes。
- 量化/分形/trans 的还原规则见 [golden-generation.md](../golden-generation.md) §2；有 ABI 时按 `abi_crosswalk`，无 ABI 时逻辑 bytes 即事实源（[launcher-development.md](../launcher/launcher-development.md)）。

### §7 同步与生命周期
- 每行一个依赖：producer → consumer，填原语（`asc_sync_pipe`/`asc_sync_notify/wait`/`asc_lock/unlock`/`asc_sync_block_arrive/wait`）、event/mutex ID、首轮/循环复用/尾轮 drain。
- 成对闭合、同 ID、事件禁 6/7（[sync-and-lifecycle.md](../sync-and-lifecycle.md)）；跨核 flag 数组按槽位索引（[fusion-design.md](../fusion-design.md) §4）。

### §8 Golden 合同
- 每行一个输出：checker（独立实现）、threshold（rtol/atol）、dtype 链、累加顺序、非有限值处理、mismatch_count 语义（bitwise vs 误差门，不混用）。
- 生成与对比方法见 [golden-generation.md](../golden-generation.md)。

### §9 构建合同
- `npu_arch` 填 `dav-3510`（小写连字符，仅编译参数）；include 策略、link set、run modes 见 [build-and-toolchain.md](../build-and-toolchain.md)。

### §10 验证与性能
- 每个 case 填 shape/dtype/目标/基线/采样/精度。基线 = 对标谁（官方算子耗时或 roofline 理论值），达标线见 [performance-analysis.md](../performance-analysis.md)。

### §11 阻塞与不支持点
- 每行：point + evidence + impact + next action。**无证据时结论只能是 BLOCKED**，不得写「有条件通过」（[review-checklist.md](../review-checklist.md) 代码检视第 6 条）。

## 2. 常见填写错误

1. **支持域写成实现**：§1 的 dtype/shape 支持域应覆盖「能支持什么」，不是「本次只做了哪几个 case」。
2. **逻辑/物理 shape 不分**：§3/§6 混写分形物理 shape 和逻辑 shape，导致 ABI 与 golden 对不上。
3. **同步 owner 空着**：§3 不填 owner，§7 的依赖无法闭合，Double Buffer/跨核同步必出时序 bug。
4. **内存预算漏中间 buffer**：只算输入/输出，漏 L1 滚动 buffer、B 槽位，容量门禁失效。
5. **尾块只写「对齐处理」**：不写显式尾块公式与 padding 处理，golden 比对阶段才暴露。

## 事实源

- 填写范本：`$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/{matmul_bias_fusion, custom_op_tensor_api}/`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。
