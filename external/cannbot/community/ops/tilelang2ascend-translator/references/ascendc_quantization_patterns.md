# 量化类算子实现指南

本指南适用于 forward 中存在浮点到整数或低比特类型映射、scale/zero-point、round/clamp、量化后写回，或量化与索引更新融合的算子。它不绑定具体算子名称或固定 shape。

## 1. 先冻结量化契约

- 明确缩放、偏置/zero-point、舍入、截断或饱和、输出类型转换的精确顺序。
- 明确 scale、zero-point 的广播轴、元素数量、dtype 组合和 optional 缺省语义。
- 分别定义正常值、边界值、零 scale、负 scale、NaN、正负 Inf 的行为，不能依赖未确认的浮点到整数转换行为。
- 明确舍入模式，特别是 half-way 输入在 reference、Ascend C API 和官方实现之间是否一致。
- 输出为整数时，精度门禁应采用严格整数一致性或算子契约规定的量化误差标准，并记录依据。

## 2. 量化数据流设计

- 让 scale/zero-point 在其生命周期内常驻并复用，避免按元素或按小行重复搬入。
- 根据 API dtype 约束决定是否升到 FP32，禁止无依据地增加多次 dtype 往返。
- 对 `round + clamp + cast` 链逐段确认 API 支持、饱和语义和尾块 mask。
- 若使用 reciprocal scale 或乘法替代除法，单独验证零值、极小值、特殊值和舍入边界。
- 中间结果的 UB 物化、reload 和转换 buffer 都要纳入 live-buffer budget；能复用的 buffer 不重复申请。
- 量化计算与大范围输出复制同时存在时，分别分析计算、搬运和 launch 固定开销，避免只优化算术链。

## 3. 量化与索引写回/重排融合

- 先将量化后的逻辑元素映射到输出布局，再决定核间和核内切分，避免大量标量地址修正。
- 对动态索引写回，明确合法范围、越界处理、重复索引和写冲突语义。
- 输出需要保留未更新区域时，区分必要的完整复制和更新区域写回，避免额外中间复制。
- 任务少而每个任务的量化长度大时，评估沿连续量化维切分；任务多而每任务较小时，评估多任务批处理以摊薄固定开销。
- 如果索引写回与复制存在重叠写区域，先证明 device 顺序和 buffer 生命周期，再考虑合并 kernel 或移除 host wait。

## 4. 克隆+量化+索引写回融合：结构范式（QuantScatter 实测）

适用形态：`output = clone(input)`（全量大 tensor）+ 量化 updates + 按 indices 写回少量
行/段。克隆与更新的流量比 ≥100:1 时，**克隆段决定性能档位**，更新段可极简。

### 4.1 结构决策：双 kernel（memcpy kernel + update kernel）

- 官方参考结构（cann/ops-nn `index/quant_update_scatter`）：out-of-place = `TensorMove`
  纯 memcpy kernel + in-place update kernel（只写更新行，不碰未更新区）。
- 存在跨核写依赖（克隆段与更新行互相覆盖）→ 拆两个 kernel，同 stream 顺序 launch
  天然有序，**禁止**在两次 launch 间插入 `stream.synchronize()`（实测：host sync
  把设备侧 1.13x 拖到端到端 0.83x）。
- 单 kernel 融合（克隆时跳过更新行、再原地补写）可行但无收益，且引入跨核时序风险，
  不推荐。

### 4.2 memcpy kernel 铁律（克隆段当纯搬运做）

- 全核铺满：input 扁平化为 1D，按 chunk（8~64KB 级，按 UB 预算撑满）均分全部 AIV 核。
- 大段 DataCopy/DataCopyPad + `TQue` depth=2（或 `SetFlag/WaitFlag<HardEvent>` 对）让
  MTE2/MTE3 重叠；**禁止 <1KB 微段、禁止段间 `PipeBarrier<PIPE_ALL>`**——实测 512B
  微段 × 每段 2 次 PIPE_ALL 把等效带宽钉死 ~100GB/s（标杆 320~1556GB/s，10x 劣化）。
- count 唯一硬约束是 **32B 对齐**（int8 即 32 的倍数）：507035 先查 count 对齐，
  与段长无因果关系；int8 可借 int32 视图搬（尾块用 DataCopyPad 的 pad 能力或单独处理）。
- 串行 depth=1 TQue 也是过矫正（实测 ~40% 性能损失）——确定性修复后按 translator
  SKILL.md 步骤 4-R 回拨。

### 4.3 update kernel 铁律

- indices/scales/zp 一次性 DataCopy 进 UB（**禁止多核 `GlobalTensor::GetValue`**，
  DCache 竞态，实测多核读索引结果漂移）。
- 量化全向量化（按行 [L] 向量 op 链），无标量循环；唯一标量是 UB 内读索引。
- **float 域先 clamp(127/-128) 再取整**（`CAST_RINT` = round-half-to-even，对齐
  `torch.round`）：±inf 天然映射边界值，且避免 scale 极小导致 int32 转换溢出。
- int8 cast 链：fp32→(RINT)→int16→(RINT)→half→(RINT)→int8（官方链，饱和语义天然
  处理 ±inf，无需显式 inf 分支）；末步仅支持 half→int8 路径，无 fp32 直转。
- 只写更新行：目标偏移 = `i*axisStride + indices[i]*lastDim`，一次 DataCopyPad 写回；
  不重读 input（out-of-place 时未更新区由 memcpy kernel 负责）。
- UB 决策树（官方 6-SplitMode 思想）：updates 全量驻留 UB（小 updates）→ 逐 bs 流式
  搬入（大 batch）→ 切 ele 分趟（单段过大）；scales/zp 始终常驻。

### 4.4 host 侧红线

- 禁止把 kernel 内可完成的 dtype 预转/预处理搬到 host（每个 ATen launch 抬小 case
  地板，实测 ~22µs vs 标杆 ~11µs）；kernel 内 cast 链有问题就修 cast 链，不外迁。
- 禁止同 stream 顺序 launch 间插 `stream.synchronize()`。

### 4.5 标杆口径

- in-place 官方版（`npu_xxx_`）= 上限标杆；out-of-place 官方版（含全量拷贝）= 公平标杆。
- 纯 Python 循环 eager 参考（逐切片赋值）只能作精度 golden，禁止作性能标杆。

## 5. 常见易错点

- 不要只按测试集 shape 分配整段 UB；大输入必须按运行时 tile 处理。
- 不要只验证随机正常值；覆盖舍入边界、饱和值、尾块、非对齐长度和特殊值。
- 不要通过指针 reinterpret 隐式接受未验证的 scale/zero-point dtype；host 契约和 kernel 模板必须一致。
- 不要把单次精度通过当作确定性通过；至少分别验证单核、多核和重复执行。
- 不要把 device task time、device span 和 host 端到端时间混为同一性能指标。

## 6. 交付检查

- 量化公式、dtype、rounding、clamp、特殊值和索引写回语义均有 reference 对照。
- 每条 dtype 路由、量化参数布局、完整/尾 tile 和索引边界均有用例。
- 量化主路径和 fallback 路径均完成编译、精度、确定性和性能验证。
- 性能报告记录 case manifest、目标架构、构建产物和统计口径。
