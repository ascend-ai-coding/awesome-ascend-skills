# Scope 分析指南

本文档提供详细的 SuperKernel Scope 范围分析方法，帮助确定最优的融合范围。

---

## 什么是 Scope

SuperKernel Scope 是指使用 `superkernel_scope` 上下文管理器标记的代码范围，编译器会将这个范围内的算子融合成一个 SuperKernel。

```python
with superkernel_scope(enable, label, option):
    # 这个范围内的算子会被融合
    pass
```

---

## Scope 范围的影响

### 范围过小
- **优点**：风险小、易调试、编译快
- **缺点**：融合效果有限、性能提升不明显

### 范围过大
- **优点**：融合效果好、性能提升潜力大
- **缺点**：编译时间长、调试困难、可能引入问题

### 最优范围
- 在性能提升和稳定性之间取得平衡
- 根据模型结构和硬件特性确定
- 需要通过实验验证

---

## Scope 分析方法

### 方法一：自底向上分析

从最小的计算单元开始，逐步扩大范围。

#### 步骤 1：识别计算密集型模块

分析模型代码，找出计算密集型模块：

```python
class DecoderLayer:
    def forward(self, hidden_states, ...):
        # 1. Self-Attention（计算密集）
        attn_output = self.self_attn(hidden_states, ...)

        # 2. Add & Norm（轻量级）
        hidden_states = hidden_states + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 3. FFN 或 MoE（计算密集）
        if self.is_moe:
            ffn_output = self.moe(hidden_states)
        else:
            ffn_output = self.ffn(hidden_states)

        # 4. Add & Norm（轻量级）
        hidden_states = hidden_states + ffn_output
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
```

**计算密集型模块**：
- Self-Attention（Q、K、V 投影 + Attention 计算 + O 投影）
- MoE（专家路由 + 专家计算）
- FFN（两层线性变换）

#### 步骤 2：确定最小 Scope

从单个计算密集型模块开始：

**选项 A：仅 Attention**
```python
def forward(self, hidden_states, ...):
    # Scope 仅包含 Attention
    with superkernel_scope(enable, "attention", option):
        attn_output = self.self_attn(hidden_states, ...)

    hidden_states = hidden_states + attn_output
    # ... 其他部分不在 Scope 内
```

**选项 B：仅 MoE**
```python
def forward(self, hidden_states, ...):
    # ... Attention 部分

    # Scope 仅包含 MoE
    with superkernel_scope(enable, "moe", option):
        ffn_output = self.moe(hidden_states)

    hidden_states = hidden_states + ffn_output
    # ...
```

#### 步骤 3：逐步扩大范围

验证最小 Scope 成功后，逐步扩大：

**扩大到整个 Decoder 层**：
```python
def forward(self, hidden_states, ...):
    # Scope 包含整个 Decoder 层
    with superkernel_scope(enable, "decoder_layer", option):
        # Attention
        attn_output = self.self_attn(hidden_states, ...)
        hidden_states = hidden_states + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        # FFN/MoE
        ffn_output = self.ffn_or_moe(hidden_states)
        hidden_states = hidden_states + ffn_output
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states
```

**扩大到多个 Decoder 层**：
```python
def decode(self, input_ids, ...):
    # Scope 包含所有 Decoder 层
    with superkernel_scope(enable, "all_layers", option):
        for layer in self.layers:
            hidden_states = layer(hidden_states, ...)

    return hidden_states
```

---

### 方法二：自顶向下分析

从最大范围开始，遇到问题时缩小范围。

#### 步骤 1：尝试最大范围

```python
# 尝试将所有 Decoder 层纳入 Scope
with superkernel_scope(enable, "all_layers", option):
    for layer in self.layers:
        hidden_states = layer(hidden_states, ...)
```

#### 步骤 2：遇到问题时缩小

如果遇到编译失败、超时、精度问题：

1. **缩小到单层**：
   ```python
   for layer in self.layers:
       with superkernel_scope(enable, f"layer_{i}", option):
           hidden_states = layer(hidden_states, ...)
   ```

2. **进一步缩小到模块**：
   ```python
   # 只包含 Attention
   with superkernel_scope(enable, "attention", option):
       attn_output = self.self_attn(hidden_states, ...)
   ```

---

### 方法三：基于性能分析

使用 profiler 分析性能瓶颈，针对性地确定 Scope。

#### 步骤 1：性能分析

```python
# 启用 profiler
import torch_npu
torch_npu.npu.profile(use_npu=True)

# 运行模型
with torch_npu.npu.profile():
    output = model(input_ids)

# 查看性能报告
```

#### 步骤 2：识别瓶颈

分析 profiler 输出，找出耗时最多的模块：
- Attention 计算
- MoE 专家计算
- FFN 计算
- 其他算子

#### 步骤 3：针对性优化

将耗时最多的模块纳入 Scope：

```python
# 如果 Attention 是瓶颈
with superkernel_scope(enable, "attention", option):
    attn_output = self.self_attn(hidden_states, ...)

# 如果 MoE 是瓶颈
with superkernel_scope(enable, "moe", option):
    ffn_output = self.moe(hidden_states)
```

---

## Scope 范围决策树

```
模型类型？
├─ MoE 模型
│   ├─ MoE 是性能瓶颈？
│   │   ├─ 是 → 优先选择"仅 MoE 模块"
│   │   └─ 否 → 选择"仅 Attention 模块"
│   └─ 追求最优性能？
│       └─ 是 → 尝试"全 Decoder 层"
│
├─ 标准 Transformer
│   ├─ 首次尝试？
│   │   └─ 是 → 选择"仅 Attention 模块"
│   ├─ Attention 验证成功？
│   │   └─ 是 → 尝试扩大到"全 Decoder 层"
│   └─ 追求最优性能？
│       └─ 是 → 使用"自动分析工具"
│
└─ 其他模型
    └─ 使用"自动分析工具"或"自定义 Scope"
```

---

## 不同模型的 Scope 建议

### LLaMA 系列

**推荐 Scope**：仅 Attention 或全 Decoder 层

```python
# 选项 1：仅 Attention
class LlamaDecoderLayer:
    def forward(self, hidden_states, ...):
        with superkernel_scope(enable, "attention", option):
            attn_output = self.self_attn(hidden_states, ...)
        # ... 其他部分

# 选项 2：全 Decoder 层
class LlamaModel:
    def decode(self, input_ids, ...):
        with superkernel_scope(enable, "all_layers", option):
            for layer in self.layers:
                hidden_states = layer(hidden_states, ...)
```

### DeepSeek-V3 系列（MoE）

**推荐 Scope**：仅 MoE 或 Attention + MoE

```python
# 选项 1：仅 MoE
class DeepSeekDecoderLayer:
    def forward(self, hidden_states, ...):
        # Attention 不在 Scope 内
        attn_output = self.self_attn(hidden_states, ...)
        hidden_states = hidden_states + attn_output

        # Scope 仅包含 MoE
        with superkernel_scope(enable, "moe", option):
            ffn_output = self.moe(hidden_states)

        hidden_states = hidden_states + ffn_output
        return hidden_states

# 选项 2：Attention + MoE
class DeepSeekDecoderLayer:
    def forward(self, hidden_states, ...):
        with superkernel_scope(enable, "decoder_layer", option):
            # Attention
            attn_output = self.self_attn(hidden_states, ...)
            hidden_states = hidden_states + attn_output

            # MoE
            ffn_output = self.moe(hidden_states)
            hidden_states = hidden_states + ffn_output

        return hidden_states
```

### Qwen 系列

**推荐 Scope**：仅 Attention

```python
class QwenDecoderLayer:
    def forward(self, hidden_states, ...):
        with superkernel_scope(enable, "attention", option):
            attn_output = self.self_attn(hidden_states, ...)

        hidden_states = hidden_states + attn_output
        # ... 其他部分
```

---

## Scope 验证方法

### 验证步骤

1. **编译验证**：
   - 检查是否编译成功
   - 查看编译日志中的 SuperKernel 信息
   - 确认 Scope 范围是否正确识别

2. **功能验证**：
   - 对比启用前后的输出
   - 精度差异应该为 0

3. **性能验证**：
   - 记录 Decode 单步耗时
   - 计算性能提升比例
   - 对比不同 Scope 策略的性能

### 验证脚本

```bash
# 1. 禁用 SuperKernel，建立基线
sed -i 's/enable_superkernel: True/enable_superkernel: False/' config.yaml
bash infer.sh > baseline.log

# 2. 启用 SuperKernel，测试性能
sed -i 's/enable_superkernel: False/enable_superkernel: True/' config.yaml
bash infer.sh > optimized.log

# 3. 对比结果
python scripts/compare_performance.py baseline.log optimized.log
```

---

## 常见问题

### Q1: 如何判断 Scope 范围是否合适？

**判断标准**：
1. 编译成功，无超时
2. 功能正确，精度一致
3. 性能提升明显（>10%）
4. 调试难度可接受

### Q2: Scope 范围可以跨越多个方法吗？

**可以**，但需要注意：
- 确保所有方法都在同一个执行路径上
- 避免包含动态控制流（if/for）
- 确保所有算子都支持 SuperKernel

### Q3: 如何处理 Prefill 和 Decode 的 Scope 差异？

**建议**：
- Prefill 阶段不启用 SuperKernel
- Decode 阶段根据分析结果启用
- 使用 `is_prefill` 标志控制

```python
with superkernel_scope(enable and not is_prefill, label, option):
    # 只在 Decode 阶段启用
    pass
```

---

## 自动分析工具

### 工具原理

自动分析工具通过以下步骤确定最优 Scope：

1. **性能分析**：使用 profiler 分析各模块耗时
2. **范围枚举**：尝试不同的 Scope 组合
3. **性能对比**：对比各组合的性能提升
4. **推荐方案**：选择性能最优且稳定的方案

### 使用方法

在 Skill 的第二步选择"使用自动分析工具"，系统会启动 subagent 执行分析。

**注意**：
- 分析过程耗时较长（5-10 分钟）
- 需要模型能够正常运行
- 结果仅供参考，需要人工验证

---

## 最佳实践

1. **从小到大**：首次尝试从最小 Scope 开始
2. **逐步验证**：每次扩大范围后都进行验证
3. **记录结果**：记录每次尝试的配置和性能数据
4. **参考案例**：查看相似模型的 Scope 配置
5. **性能优先**：在稳定性保证的前提下追求性能
6. **问题排查**：遇到问题先缩小范围，定位根因

---

## 参考资源

- Scope 模板：`../resources/scope-templates/`
- 性能基线指南：`performance-baseline-guide.md`
- 调试指南：`debugging-guide.md`
- 官方文档：[SuperKernel 开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_00029.html)
