---
name: external-cannbot-model-model-infer-superkernel
description: 基于 PyTorch 框架的昇腾 NPU 模型推理 SuperKernel 适配技能。当用户需要启用 SuperKernel 算子二进制融合技术优化昇腾
  NPU 推理性能时使用此技能。触发场景包括：用户询问 SuperKernel、算子融合、二进制融合、启用 superkernel、superkernel_scope、减少任务调度开销、优化
  decode 性能等。SuperKernel 仅支持 ge_graph 模式、Atlas A3 硬件，且仅在 decode 阶段生效。
original-name: model-infer-superkernel
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# SuperKernel 适配技能

提供 SuperKernel 算子二进制融合技术的完整适配流程，包括模型分析、Scope 范围确定、代码实施和性能验证。

---

## 重要原则

- **前置条件**：必须满足 `exe_mode: "ge_graph"`、Atlas A3 硬件、PyTorch 框架
- **仅 Decode 阶段生效**：Prefill 阶段输入长度动态变化，SuperKernel 自动禁用
- **手动标记 Scope**：需要使用 `superkernel_scope` 上下文管理器标记融合范围
- **配置互斥**：不支持 `eager` 模式和 `aclgraph` 模式
- **先理解再行动**：分析模型结构后再确定 Scope 范围，避免盲目复制其他模型配置
- **组合使用**：SuperKernel 作为优化技术之一，可与其他优化技术（多流并行、融合算子等）组合使用

---

## 重要提醒

**必须完成所有四个步骤**：
1. 分析模型结构
2. 确定 Scope 范围
3. 代码编写
4. **性能测试验证**（必须执行，不可跳过）

**最终产出**：
- 优化后的代码文件（config.yaml、modeling_*.py）
- SuperKernel 优化文档（`superkernel_optimization_report.md`）

---

## 工作流程

### 第一步：分析模型结构

**目标**：理解模型架构，识别可以应用 SuperKernel 的部分

1. **读取模型代码**：
   - 找到模型实现文件（`cann-recipes-infer/models/{model_name}/models/modeling_*.py`）
   - 找到配置文件（`cann-recipes-infer/models/{model_name}/config.yaml`）
   - 找到 runner 文件（`cann-recipes-infer/models/{model_name}/models/runner_*.py`）

2. **识别模型架构特征**：
   - 是否有 Attention 层（Self-Attention、Cross-Attention）
   - 是否有 MoE 层（专家路由、专家计算）
   - 是否有其他计算密集型模块（FFN、LayerNorm 等）
   - Prefill 和 Decode 是否已分离

3. **检查当前配置**：
   - `exe_mode` 是否为 `ge_graph`（必须）
   - `enable_superkernel` 当前状态
   - 硬件是否为 Atlas A3
   - 是否已启用图模式（`torch.compile`）

4. **识别潜在问题**：
   - 动态 shape 问题（KV Cache 是否预分配）
   - 控制流问题（是否有 Python if/for）
   - 不支持的算子（Tiling 下沉算子）

**产出**：
- 模型结构分析（内部记录，不输出文件）
- 可应用 SuperKernel 的模块列表
- 潜在问题清单

---

### 第二步：确定 Scope 范围

**目标**：确定 SuperKernel 的融合范围

使用 AskUserQuestion 提供以下选项：

| 选项 | 描述 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| **仅 Attention 模块** | 只将 Attention 计算纳入 SuperKernel scope | 首次尝试、模型瓶颈在 Attention | 风险小、易调试 |
| **仅 MoE 模块** | 只将 MoE 专家计算纳入 SuperKernel scope | MoE 架构模型 | 针对性强、效果明显 |
| **全模型 Decoder 层** | 将整个 Decoder 层纳入 SuperKernel scope | 模型结构简单、已有成功案例 | 融合范围最大、性能提升潜力最高、调试难度较大 |
| **使用自动分析工具** | 启动 subagent 分析最优 Scope 划分 | 追求最优性能 | 基于性能分析自动推荐、耗时较长（5-10分钟） |
| **自定义 Scope** | 手动指定需要融合的模块 | 有特殊需求或已知最优配置 | 灵活性高 |

**执行内容**：
- 根据用户选择，生成对应的 Scope 配置
- 如果选择"自动分析工具"，启动 subagent 执行性能分析
- 生成 Scope 标记代码片段
- 确定需要修改的文件和具体位置

**产出**：
- Scope 配置方案（内部记录，不输出文件）
- 需要修改的代码位置和具体改动

> 详细的 Scope 分析方法见 `references/scope-analysis-guide.md`

---

### 第三步：代码编写

**目标**：实施 SuperKernel 适配代码

**重要**：这一步只修改代码，不输出任何中间文档。

#### 3.1 配置文件修改

修改 `cann-recipes-infer/models/{model_name}/config.yaml`：

```yaml
# 确保以下配置
exe_mode: "ge_graph"              # 必须是 ge_graph
model_config:
  enable_superkernel: True        # 启用 SuperKernel
  enable_multi_streams: False     # 根据需求配置
  enable_cache_compile: False     # 根据需求配置
```

#### 3.2 模型代码修改

在 `cann-recipes-infer/models/{model_name}/models/modeling_*.py` 中：

```python
# 1. 导入上下文管理器
from executor.utils import superkernel_scope

# 2. 在 decode 方法中添加 Scope 标记
class YourModel:
    def decode(self, input_ids, ...):
        is_prefill = False  # decode 阶段

        # 根据选择的 Scope 策略添加
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            label="decode_layers",  # 或 "attention_only", "moe_only"
            option="stream-fusion=1"  # 编译选项
        ):
            for decoder_layer in self.layers:
                # 运算逻辑
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    ...
                )

        return hidden_states
```

**Scope 标记位置示例**：

- **仅 Attention**：在 Attention 模块的 forward 方法中
- **仅 MoE**：在 MoE 层的 forward 方法中
- **全 Decoder 层**：在整个 Decoder 层循环外层

**产出**：
- 修改后的代码文件（config.yaml、modeling_*.py 等）

---

### 第四步：性能测试验证（必须执行）

**目标**：验证 SuperKernel 适配的正确性和性能提升

**重要**：这一步必须执行，不可跳过。完成代码修改后立即进行性能测试。

#### 4.1 建立性能基线

1. **临时禁用 SuperKernel**：
   ```bash
   cd cann-recipes-infer/models/{model_name}
   # 备份当前配置
   cp config.yaml config.yaml.superkernel
   # 禁用 SuperKernel
   sed -i 's/enable_superkernel: True/enable_superkernel: False/' config.yaml
   ```

2. **运行基线测试**：
   ```bash
   bash infer.sh 2>&1 | tee baseline.log
   ```

3. **记录基线性能**：
   - Decode 单步耗时（ms）
   - 吞吐量（tokens/s）
   - 从日志中提取关键性能指标

#### 4.2 测试优化版本

1. **恢复 SuperKernel 配置**：
   ```bash
   # 恢复 SuperKernel 配置
   cp config.yaml.superkernel config.yaml
   ```

2. **运行优化测试**：
   ```bash
   bash infer.sh 2>&1 | tee optimized.log
   ```

3. **记录优化性能**：
   - Decode 单步耗时（ms）
   - 吞吐量（tokens/s）
   - 从日志中提取关键性能指标

#### 4.3 性能对比分析

计算性能提升：
```
性能提升 = (基线耗时 - 优化后耗时) / 基线耗时 × 100%
吞吐量提升 = (优化后吞吐量 - 基线吞吐量) / 基线吞吐量 × 100%
```

#### 4.4 生成优化报告

**必须生成**：创建 `superkernel_optimization_report.md`，包含：

```markdown
# SuperKernel 优化报告

## 优化配置

- 模型：{model_name}
- Scope 策略：{scope_strategy}
- 优化日期：{date}

## 代码修改

### 配置文件修改
- 文件：`config.yaml`
- 修改内容：启用 `enable_superkernel: True`

### 模型代码修改
- 文件：`cann-recipes-infer/models/modeling_*.py`
- 修改内容：添加 SuperKernel scope 标记
- Scope 范围：{scope_description}

## 性能测试结果

### 基线性能（SuperKernel 禁用）
- Decode 单步耗时：{baseline_latency} ms
- 吞吐量：{baseline_throughput} tokens/s

### 优化后性能（SuperKernel 启用）
- Decode 单步耗时：{optimized_latency} ms
- 吞吐量：{optimized_throughput} tokens/s

### 性能提升
- 延迟降低：{latency_improvement}%
- 吞吐量提升：{throughput_improvement}%

## 结论

{conclusion}

## 使用说明

优化后的模型配置已保存，可直接使用：
\`\`\`bash
cd cann-recipes-infer/models/{model_name}
bash infer.sh
\`\`\`
```

**产出**：
- 性能测试日志（baseline.log、optimized.log）
- **SuperKernel 优化报告**（`superkernel_optimization_report.md`）

> 详细的性能基线建立方法见 `references/performance-baseline-guide.md`

---

## 最终交付物

完成所有四个步骤后，必须提供：

1. **优化后的代码文件**：
   - `cann-recipes-infer/models/{model_name}/config.yaml`（已启用 SuperKernel）
   - `cann-recipes-infer/models/{model_name}/models/modeling_*.py`（已添加 Scope 标记）

2. **SuperKernel 优化报告**：
   - `superkernel_optimization_report.md`
   - 包含完整的性能对比数据和优化说明

**不需要输出**：
- 中间分析文档（superkernel_analysis.md）
- Scope 规划文档（superkernel_scope_plan.md）
- 其他临时文件

---

## Scope 选择指南

### 快速决策树

```
模型类型？
├─ MoE 模型 → 优先选择"仅 MoE 模块"
├─ 标准 Transformer → 优先选择"仅 Attention 模块"
└─ 简单模型 + 有成功案例 → 可选择"全 Decoder 层"

首次尝试？
└─ 是 → 选择"仅 Attention 模块"（风险最小）

追求最优性能？
└─ 是 → 选择"使用自动分析工具"

已知最优配置？
└─ 是 → 选择"自定义 Scope"
```

### Scope 模板

详细的 Scope 模板见 `resources/scope-templates/` 目录：
- `attention-only.md` - 仅 Attention 的 Scope 模板
- `moe-only.md` - 仅 MoE 的 Scope 模板
- `full-model.md` - 全模型 Scope 模板

---

## 配置检查清单

启用 SuperKernel 前，请确保：

- [ ] `exe_mode` 设置为 `ge_graph`（不能是 `eager` 或 `aclgraph`）
- [ ] 硬件为 Atlas A3 系列
- [ ] 框架为 PyTorch
- [ ] 模型已区分 prefill 和 decode 阶段
- [ ] 已正确标记 SuperKernel scope

---

## 参考文档

### 内部文档
- `references/scope-analysis-guide.md` - Scope 分析详细指南
- `references/performance-baseline-guide.md` - 性能基线建立指南
- `resources/scope-templates/` - Scope 模板

### 外部文档
- [官方 SuperKernel 开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_00029.html)
- [PyTorch 图模式使用指南](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html)
- [DeepSeek-R1 decode 性能优化](../../docs/models/deepseek_r1/deepseek_r1_decode_optimization.md)
- [SuperKernel 易用性指南](../../usability_docs/superkernel_usability_guide.md)

---

## 最佳实践

1. **首次尝试**：选择"仅 Attention 模块"，风险最小
2. **逐步扩大**：验证成功后再扩大 Scope 范围
3. **性能对比**：每次修改后都进行性能对比
4. **文档记录**：记录每次尝试的配置和结果
5. **问题排查**：遇到问题先查看检查清单和常见问题
6. **寻求帮助**：复杂问题可以调用其他 skill 或查阅详细文档
