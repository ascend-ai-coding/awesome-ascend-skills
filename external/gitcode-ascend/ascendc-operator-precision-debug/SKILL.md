---
name: external-gitcode-ascend-ascendc-operator-precision-debug
description: AscendC 算子精度问题调试与根因定位。当算子精度测试失败（allclose 不通过、结果偏差、输出全零/NaN 等）时使用。流程：误差分布分析
  → 代码易错点审查 → 实验隔离 → printf/DumpTensor 插桩 → 修复验证。关键词：精度调试、精度问题、结果不一致、误差定位、allclose
  失败、输出偏差、NaN、全零、precision debug。
original-name: ascendc-operator-precision-debug
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# AscendC 算子精度调试

按「由浅入深」五阶段定位根因：**先看数据分布，再查代码易错点，然后实验隔离，最后插桩定位。**

```
Phase 1: 误差分析 → Phase 2: 代码审查 → Phase 3: 实验隔离 → Phase 4: 插桩定位 → Phase 5: 修复验证
```

---

## Phase 1：误差分析

**原则：先看数据，再看代码。** 先搞清楚「错在哪、错多少、错成什么样」。

收集失败用例的 shape、dtype、MaxAbsErr/MeanAbsErr/CosineSim，然后基于 [`scripts/debug_precision_template.py`](scripts/debug_precision_template.py) 创建 `csrc/ops/<op_name>/test/debug_<op_name>_precision.py`（替换占位符后运行），自动分析：

1. **误差统计**：MaxAbsErr、MeanAbsErr、MaxRelErr
2. **首个错误元素**：多维坐标 + 线性下标 + NPU 值 vs 参考值
3. **错误分布**：错误元素数量/占比、错误间隔是否呈周期性
4. **特殊值**：输出是否全零、含 NaN/Inf
5. **自动对照**：固定输入 vs 随机输入、缩小 shape 二分

### 误差特征 → 初步判断

| 现象 | 最可能原因 | 下一步 |
|------|-----------|--------|
| FP16 失败，FP32 通过 | **未升精度到 FP32 计算** | Phase 2 查 Cast |
| 输出全零 | CopyOut 未执行 / GM 偏移错 | Phase 2 查 CopyOut |
| 输出含 NaN/Inf | 除零 / log 负数 / 溢出 | Phase 2 查 Compute |
| 全部偏差，CosineSim≈1 | 系统性精度损失 | Phase 2 查升精度 |
| 周期性/条纹状错误 | tile 边界 / 搬运偏移 | Phase 3 实验 |
| 仅尾部元素错 | 尾 tile 长度 / 对齐 | Phase 2 查尾 tile |
| 多次运行结果不同 | 异步同步不足 | Phase 3 实验 B |
| 小 shape 过、大 shape 挂 | 多核/tiling 边界 | Phase 3 实验 A |
| 固定输入过、随机挂 | 地址/stride/偏移错 | Phase 3 实验 C |

---

## Phase 2：代码审查

**MANDATORY**：读取 `op_host/<op_name>.cpp`、`op_kernel/<op_name>.cpp`、`design.md`（若存在），按以下清单由浅入深排查。

### 第一层：基本正确性（最高频）

- [ ] **FP16/BF16 未升精度**：Compute 中半精度是否先 `Cast` 到 FP32 计算再 `Cast` 回？**这是最高频精度 bug。**
- [ ] **计算公式错误**：API 调用序列与设计文档/PyTorch 逐步对照——运算顺序、标量符号、是否缺步骤。
- [ ] **GM 偏移单位混淆**：`xGm[progress * tileLength]` 是元素偏移，不要多乘 `sizeof(T)`。
- [ ] **tileLength vs curTileLength**：偏移用 `tileLength`，计算/搬运用 `curTileLength`（尾 tile 可能更小）。

### 第二层：搬运与对齐

- [ ] **DataCopyPad copyLen**：`DataCopyExtParams` 的 copyLen 是字节数 = `curTileLength * sizeof(T)`。
- [ ] **尾 tile 对齐**：尾 tile 不满足 32B 对齐时，`alignedTailLen` 计算及使用是否正确。
- [ ] **多输入偏移不一致**：多输入 tensor shape 不同时（如 RoPE 的 x vs cos/sin），各自的偏移计算是否正确。

### 第三层：Tiling 与多核

- [ ] **Host/Kernel tiling 不一致**：同一符号（如 `tileLength`）在 host 和 kernel 中含义是否一致。
- [ ] **核间边界重叠/遗漏**：formerNum × formerLength + tailNum × tailLength 是否恰好覆盖全部数据。
- [ ] **bufferCoefficient 错误**：与设计文档 UB 分配表核对，错误的系数会导致 tileLength 偏差。

### 第四层：API 陷阱

- [ ] **ReduceSum/Max 修改源数据**：归约可能改写源 tensor，后续若复用需先 `Adds(backup, src, 0.0f, len)` 备份。
- [ ] **AllocTensor/FreeTensor 未配对**：与 EnQue/DeQue 需严格配对，否则缓冲区泄漏。
- [ ] **向量长度参数**：AscendC 向量 API 长度是元素个数，非字节数。

### 第五层：边界情况

- [ ] **除零 / 定义域越界**：Div、Reciprocal 防零；Ln 要求正数；Sqrt 要求非负。
- [ ] **tiling 整数溢出**：乘法是否可能溢出 int32？建议 int64_t。

**检查点**：输出审查报告——**疑似问题列表（按可能性排序）**。若已锁定根因，跳到 Phase 5；否则进入 Phase 3。

---

## Phase 3：实验隔离

Phase 2 无法直接锁定根因时，通过控制变量实验缩小范围。**每次只改一个变量。**

### 实验 A：block_dim → 1（多核隔离）

在 op_host 临时硬编码 `blockDim = 1`，重编译测试。可配合缩小 shape。

| 结果 | 结论 |
|------|------|
| 单核过、多核挂 | 核间问题：GM 区间重叠 / tiling 映射 / 核间同步 |
| 单核也挂 | 非多核问题 → 实验 B |

### 实验 B：PipeBarrier\<PIPE_ALL\>（同步隔离）

将 kernel Process 中所有同步临时替换为 `AscendC::PipeBarrier<PIPE_ALL>()`（CopyIn / Compute / CopyOut 之间各加一个）。

| 结果 | 结论 |
|------|------|
| 全屏障后过 | 核内同步不足 → 逐步恢复细粒度同步定位 |
| 仍失败 | 非同步问题 → 实验 C |

> `PIPE_ALL` 仅用于实验隔离，**绝不可作为最终方案**。

### 实验 C：固定/规律输入（地址隔离）

分别用全 1、等差序列（`torch.arange`）、随机输入测试。

| 结果 | 结论 |
|------|------|
| 全 1 过、等差/随机挂 | 地址/偏移/stride 错误（常数输入掩盖了偏移问题） |
| 全都挂 | 计算逻辑或全局 tiling 错误 |
| 全都过 | 特定数值范围触发精度问题 → 查边界值/极值 |

### 实验 D：缩小 shape（边界隔离）

`shape=(32,)` → `(tileLength,)` → `(tileLength*2,)` → 原始 shape，定位恰好开始失败的分界点，反推 tile/核边界。

### 首错下标 + tiling 反推

```
首错线性下标 → 第几个 tile → 哪个核 → 该核 GM 起始偏移 → 搬运预期字节数
```

周期 = tileLength → 搬运/偏移问题；周期 = 向量宽度 → 计算流程问题；与核边界对齐 → 多核/offset 问题。

---

## Phase 4：插桩定位

问题范围已收敛到某阶段/某 tile 后，用 `AscendC::printf` 和 `AscendC::DumpTensor` 精确定位。

### 核心规则

1. **仅 0 核打印**：每个核计算逻辑一致时，加 `if (AscendC::GetBlockIdx() == 0)` 减少输出量。
2. **同步后再读**：在 `DeQue` / `PipeBarrier` 之后才能读 `LocalTensor`，否则读到未完成搬运的脏数据。
3. **FP16 先转 float**：`AscendC::printf("v=%.6f\n", static_cast<float>(tensor.GetValue(idx)));`，直接打 half 会乱码。
4. **用 desc 区分阶段**：DumpTensor 的 desc 参数（0=CopyIn 后, 1=Compute 中间, 2=CopyOut 前）。
5. **小量起步**：DumpTensor 的 dumpSize 从小值开始，过大会导致缓冲满或截断。

### printf vs DumpTensor 选择

| 场景 | 工具 |
|------|------|
| 标量、分支判断、单个下标 | `AscendC::printf` |
| 连续一段 tensor 快速扫 | `AscendC::DumpTensor(tensor, desc, dumpSize)` |
| 全量逐元素对比 | **不在 kernel 内做** — Host 读 GM + Python 脚本 |

### 插桩策略

在 Compute 函数内 DeQue 之后，逐步骤插桩，与 Python 侧用相同输入手算的中间结果逐步对比。**第一个出现偏差的步骤即为根因所在。**

```cpp
// 示意：0 核、第 0 个 tile
if (AscendC::GetBlockIdx() == 0 && progress == 0) {
    AscendC::printf("[step1] tmp[0]=%.6f\n", static_cast<float>(tmp.GetValue(0)));
}
```

---

## Phase 5：修复验证

### 常见修复模式

| 根因 | 修复 |
|------|------|
| FP16 未升精度 | 添加 Cast(fp16→fp32) + 计算 + Cast(fp32→fp16) |
| GM 偏移错 | 修正偏移公式（元素 vs 字节） |
| 尾 tile 长度错 | 计算/搬运用 curTileLength，偏移用 tileLength |
| tiling 参数错 | 修正 host 端 tiling 计算 |
| 同步缺失 | 添加正确的 EnQue/DeQue 或 PipeBarrier |
| ReduceSum 覆盖源 | 先 Adds 备份再 ReduceSum |
| 搬运长度错 | 修正 DataCopyExtParams 的 copyLen |

### 修复后

1. **移除所有调试插桩**（printf/DumpTensor），或用 `#ifdef DEBUG_PRECISION` 包裹
2. 重新编译安装
3. 运行原失败用例 + 完整精度测试
4. 仍失败 → 回到 Phase 1（最多 3 轮），3 轮后仍失败则报告用户

### 输出要求（MANDATORY）

调试完成后 **MUST** 在对话中展示：问题摘要、根因分析、修复内容、验证结果、≥2 条关键经验。**NEVER** 仅回复「已修复」。

---

## 典型案例（按需加载）

定位到疑似根因后，加载对应案例了解完整排查过程：

| 误差现象 | 案例文件 | 何时加载 |
|----------|---------|---------|
| FP16 挂 FP32 过，全部偏差 | [`examples/fp16-no-upcast.md`](examples/fp16-no-upcast.md) | 怀疑升精度缺失 |
| 首错在 tile 边界，周期 = tileLength | [`examples/gm-offset-error.md`](examples/gm-offset-error.md) | 怀疑 GM 偏移错误 |
| 仅尾部少量元素错 | [`examples/tail-tile-misalign.md`](examples/tail-tile-misalign.md) | 怀疑尾 tile 处理 |
| block_dim=1 过，多核挂 | [`examples/multicore-tiling-overlap.md`](examples/multicore-tiling-overlap.md) | 怀疑核间 tiling |
| 多次运行结果不同 | [`examples/async-sync-missing.md`](examples/async-sync-missing.md) | 怀疑同步缺失 |

> **不要一次性加载所有案例。** 仅在误差特征匹配时加载对应案例。

---

## 反模式（NEVER）

- **NEVER** 不分析误差分布就直接改代码
- **NEVER** 在 kernel 中 printf 循环打全量 tensor — 用 DumpTensor 或 Host 侧对比
- **NEVER** 多核同时大量打印 — 加 `GetBlockIdx() == 0` 仅 0 核打印
- **NEVER** 在未同步位置读 LocalTensor — 必须在 DeQue/PipeBarrier 之后
- **NEVER** 用 `PIPE_ALL` 作为最终修复 — 仅用于实验隔离
- **NEVER** 修复后不移除调试代码
- **NEVER** 仅修复已知失败用例而不跑完整精度测试
- **NEVER** 超过 3 轮仍失败时继续尝试 — 应报告用户
