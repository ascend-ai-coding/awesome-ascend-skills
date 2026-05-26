---
name: external-cannbot-model-model-infer-prefetch
description: 基于 PyTorch 框架的昇腾 NPU 模型推理权重预取优化技能。为模型添加 torch_npu.npu_prefetch 特性以优化推理性能。触发场景：profiling
  显示 MatMul/QBMM/GMM 算子存在 memory-bound 热点、需要为模型添加权重预取、将 prefetch 模式迁移到新模型。
user-invocable: true
original-name: model-infer-prefetch
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# Prefetch 预取优化技能

提供手动指定和自动化分析两种方式确定预取位置,覆盖方案设计、大小计算、代码实现和性能验证。

---

## 重要原则

- **前置条件**:模型必须在 NPU 上运行且已导入 `torch_npu`
- **仅在 memory-bound 热点时使用**:必须通过 profiling 确认存在 memory-bound 瓶颈
- **必须有安全的依赖窗口**:前序算子不能是 memory-bound,否则会争抢带宽
- **始终提供开关保护**:所有 prefetch 必须用 `enable_prefetch` 开关保护,默认 `False`
- **先保守后激进**:`max_size` 从保守值开始,根据 profiling 结果逐步调整
- **单点验证原则**:先验证单个目标算子,成功后再扩展到多个位置
- **图模式兼容性**:在图模式下,依赖节点必须与目标算子在逻辑时序上匹配

---

## 工作流程

### 第一步:方案设计

1. **确定预取位置**:选择手动指定或自动化分析方式
   - 手动指定:用户提供目标算子和依赖窗口
   - 自动化分析:调用 `npu-roofline-analysis` 从 profiling 数据中识别候选位置
2. **确定预取大小**:根据权重张量维度和数据类型计算 `max_size`
3. **设计改造方案**:
   - 需要修改哪些文件
   - 如何传递 `enable_prefetch` 开关
   - 在哪里插入 `torch_npu.npu_prefetch` 调用
   - `max_size` 和 `dependency` 的具体取值
4. **输出设计方案文档**:以 Markdown 格式呈现

### 第二步:方案确认

等待用户确认方案后再进行开发。有修改意见则返回第一步。

### 第三步:实施开发

按照确认的设计方案逐步实施代码修改:
- 新增或贯通 `enable_prefetch` 开关
- 插入条件保护的 `torch_npu.npu_prefetch` 调用
- 配置 `max_size` 和 `dependency` 参数

### 第四步:验证测试

由 Agent 实际执行以下测试,记录真实结果:
- **编译验证**:检查代码语法和运行时错误
- **功能验证**:对比 prefetch 前后输出一致性
- **性能验证**:运行 profiling,对比目标算子耗时、关键路径、依赖窗口
- **测试报告**:整理前后对比数据,评估优化效果

**重要**:必须实际运行测试并生成性能优化报告,包括:
1. Baseline 性能数据 (enable_prefetch=False)
2. 优化后性能数据 (enable_prefetch=True)
3. 性能对比和改善百分比
4. 验证检查清单结果
5. 后续调优建议

---

## 预取位置确定

在开始实施前先确定使用哪种方式。如果用户未指定,优先采用自动化分析。

| 方式 | 输入要求 | 输出结果 | 适用场景 |
|------|---------|---------|---------|
| **手动指定位置** | 用户提供目标算子和依赖窗口 | 直接进入大小确定阶段 | 已有明确优化目标,熟悉模型结构 |
| **自动化分析** | profiling 数据或模型代码 | 候选位置列表 + 理论预取大小 | 需要从数据中发现机会,不确定优化点 |

### 手动指定方式

用户直接提供:
- **目标算子**:需要优化的 memory-bound 算子(如 MatMul, QBMM, GMM)
- **依赖窗口**:目标算子的前序非 memory-bound 算子
- **权重信息**:权重张量的维度和数据类型

直接跳转到"预取大小确定"章节。

### 自动化分析方式

通过调用 `npu-roofline-analysis` skill 和分析模型代码自动识别候选位置。

#### 自动化分析流程

```
开始
  │
  ├─→ 步骤 1:调用 npu-roofline-analysis
  │       ↓
  │   获取 memory-bound top 10 算子列表
  │       ↓
  ├─→ 步骤 2:对每个 memory-bound 算子
  │       ├── 分析前序算子
  │       ├── 判断前序是否访存密集
  │       │
  │       ├─→ 前序是访存密集 ──→ 无预取空间 ──→ 跳过该算子
  │       │
  │       └─→ 前序非访存密集 ──→ 有预取空间 ──→ 标记为候选
  │               ↓
  ├─→ 步骤 3:计算理论预取大小
  │       ├── 获取权重张量维度
  │       ├── 获取数据类型(bf16/int8)
  │       └── 计算 max_size = 维度 × 字节数
  │           ↓
  └─→ 输出:候选位置列表 + 理论预取大小
```

#### 自动化分析决策表

| 步骤 | 检查项 | 结果 | 下一步 |
|------|--------|------|--------|
| 1 | 调用 `npu-roofline-analysis` | 获取 memory-bound top 10 算子 | 进入步骤 2 |
| 2 | 调用 `npu-roofline-analysis`分析每个算子的前序算子（前2-3个） | 判断前序是否访存密集 | 进入步骤 3 |
| 3a | 前序算子是访存密集(大 MatMul/通信)但其输出可用 | 有预取空间(使用前序输出作为依赖窗口) | 计算理论预取大小 |
| 3b | 前序算子非访存密集(LayerNorm/ROPE/SwiGLU/小算子) | 有预取空间(使用前序输出作为依赖窗口) | 计算理论预取大小 |
| 3c | 前序算子是通信或无合适窗口 | 无预取空间 | 跳过该算子 |
| 4 | 计算权重大小 | 理论 max_size | 输出候选列表 |

#### 前序算子访存密集判断标准

**访存密集定义**:指 memory-bound 算子,即内存带宽成为瓶颈的算子。

| 算子类型 | 是否访存密集(memory-bound) | 说明 |
|---------|---------------------------|------|
| **大型 MatMul, BatchMatMul, QBMM, GMM** | **是** | 权重搬运密集,memory-bound,但其**输出可作为依赖窗口** |
| **通信算子(send, recv, all_reduce)** | **是** | 带宽密集,不适合作为依赖窗口 |
| **大型数据搬运(大 transpose, 大 concat)** | **是** | 数据搬运量大,会争抢带宽 |
| LayerNorm, RMSNorm | **否** | 虽是访存类算子,但不是 memory-bound,**可作为依赖窗口** |
| ROPE, cast, reshape, 小 transpose | **否** | 计算或搬运量小,**可作为依赖窗口** |
| SwiGLU, GELU, Silu, npu_dequant_swiglu_quant | **否** | 计算密集,**可作为依赖窗口** |
| router topk, dispatch, combine | **否** | 计算密集,**可作为依赖窗口** |
| router.classifier(小型 MatMul) | **否** | 权重较小,**输出可作为依赖窗口** |

#### 自动化分析示例

**场景**:LongCat-Flash 模型优化

1. **调用 npu-roofline-analysis**:
   - 识别 memory-bound top 10:gate_up_proj(MatMul), down_proj(MatMul), q_a_proj(MatMul), q_b_proj(MatMul), kv_a_proj_with_mqa(MatMul), router.classifier(MatMul), GMM1, GMM2, ...

2. **分析 gate_up_proj 前序算子**:
   - 前序:o_proj(大型 MatMul, memory-bound)
   - 判断:访存密集 ✗
   - 结论:但实际 o_proj 输出可作为依赖窗口(o_proj 执行完成后带宽空闲)
   - **实际使用**:在 o_proj 完成时预取 gate_up_proj 权重 ✓

3. **分析 down_proj 前序算子**:
   - 前序:SwiGLU(计算密集,非 memory-bound)
   - 判断:非访存密集 ✓
   - 结论:可预取,依赖窗口 = SwiGLU 输入 x
   - **实际使用**:在 gate_up_proj 输入时预取 down_proj 权重 ✓

4. **分析 q_a_proj 前序算子**:
   - 前序:RMSNorm(虽是访存类算子,但非 memory-bound)
   - 判断:非访存密集 ✓
   - 结论:可预取,依赖窗口 = down_proj 输出
   - **实际使用**:在 down_proj 完成时预取下一层 q_a_proj 权重 ✓

5. **分析 router.classifier 前序算子**:
   - 前序:router 输入(来自 recv 或 o_proj)
   - 判断:非访存密集 ✓
   - 结论:可预取,依赖窗口 = o_proj 输出
   - **实际使用**:在 o_proj 完成时预取 router.classifier 权重 ✓

6. **计算理论预取大小**:
   - gate_up_proj:`hidden_size × intermediate_size × 2 × 2` 字节(bf16)
   - 保守取值:理论值的 50%

7. **输出候选列表**:
   - 候选 1:gate_up_proj, 依赖窗口 = o_proj 输出, max_size = 计算值
   - 候选 2:down_proj, 依赖窗口 = gate_up_proj 输入, max_size = 计算值
   - 候选 3:q_a_proj(下一层), 依赖窗口 = down_proj 输出, max_size = 18 MB
   - 候选 4:router.classifier, 依赖窗口 = o_proj 输出, max_size = 18 MB
   - ...

> 详细的 roofline 分析方法参见 `npu-roofline-analysis` skill

---

## 预取大小确定

根据权重张量维度和数据类型计算 `max_size`,遵循"先保守后激进"原则。

### 计算公式

**基本公式**:
```
max_size = 权重维度乘积 × 数据类型字节数 × 保守系数
```

**数据类型字节数**:
- `bf16` / `fp16`: 2 字节
- `int8`: 1 字节
- `fp32`: 4 字节

**保守系数**:
- 首次尝试:0.5(理论值的 50%)
- 验证成功后:逐步提升到 0.7 → 1.0

### 不同算子的计算示例

| 算子类型 | 权重维度 | 数据类型 | 理论 max_size | 保守 max_size(50%) |
|---------|---------|---------|--------------|-------------------|
| **MatMul(q_proj)** | `hidden_size × hidden_size` | bf16 | `H × H × 2` | `H × H` |
| **MatMul(o_proj)** | `hidden_size × hidden_size` | bf16 | `H × H × 2` | `H × H` |
| **MatMul(gate_up_proj)** | `hidden_size × intermediate_size × 2` | bf16 | `H × I × 4` | `H × I × 2` |
| **MatMul(down_proj)** | `intermediate_size × hidden_size` | bf16 | `I × H × 2` | `I × H` |
| **QBMM** | `hidden_size × hidden_size` | int8 | `H × H` | `H × H / 2` |
| **GMM(MoE)** | `hidden_size × intermediate_size × 2 / moe_tp_size × experts_per_rank` | bf16/int8 | 见下方 | 见下方 |

### MoE GMM 预取大小计算

```python
# 数据类型字节数
dtype_bit = 1 if quant_mode == "w8a8" else 2  # int8: 1 字节, bf16: 2 字节

# GMM1(gate_up_proj)预取大小
gmm1_prefetch_size = hidden_size * intermediate_size * 2 * dtype_bit // moe_tp_size * experts_per_rank // 2

# GMM2(down_proj)预取大小
gmm2_prefetch_size = hidden_size * intermediate_size * dtype_bit // moe_tp_size * experts_per_rank
```

### 调整策略

| 场景 | 调整方向 | 说明 |
|------|---------|------|
| 目标算子加速明显,依赖窗口稳定 | 增大 max_size | 从 50% → 70% → 100% |
| 目标算子加速不明显 | 保持或减小 max_size | 可能预取收益有限 |
| 依赖窗口明显退化 | 减小 max_size | 从 50% → 30% → 10% |
| 依赖窗口严重退化 | 调整位置或放弃 | 前移 prefetch 或选择其他依赖窗口 |

### LongCat-Flash 实际取值参考

| 位置 | 目标算子 | 依赖窗口 | max_size | 说明 |
|------|---------|---------|----------|------|
| Line 1244 | router classifier | o_proj | 18 MB | 固定值 |
| Line 1259 | q_a_proj | dense MLP output | 18 MB | 保守值 |
| Line 1260 | q_b_proj | dense MLP output | 36 MB | 中等值 |
| Line 1261 | kv_a_proj_with_mqa | dense MLP output | 7 MB | 保守值 |
| Line 1286 | next layer attention | down_proj | 计算值 | 动态计算 |

> 参考文件:`models/longcat-flash/models/modeling_longcat_flash.py`, `models/longcat-flash/models/ffn.py`

---

## 目标算子与依赖窗口选择

### 目标算子优先级

| 优先级 | 算子类型 | 说明 |
|--------|----------|------|
| **高** | MatMul, BatchMatMul, QBMM, GMM | 主要预取目标,权重搬运密集 |
| **高** | 大型 MLP/FFN 线性层 | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_up_proj |
| **高** | MoE 相关 MatMul | router.classifier, experts.w13_weight, experts.w2_weight |
| **高** | 融合前置大搬运算子 | 如 MLAProlog(需 roofline 支持) |
| **低** | LayerNorm, RMSNorm, ROPE | 通常作为依赖窗口,不是预取目标 |
| **低** | cast, reshape, transpose, concat | 同上 |
| **低** | router topk, dispatch, combine | 同上 |
| **低** | SwiGLU, GELU, Silu | 同上 |

### 依赖窗口选择流程

```
开始选择依赖窗口
  │
  ├─→ 找到目标算子的紧邻前驱
  │       ↓
  │   检查:是否 memory-bound?
  │       ├─→ 是 ──→ 前移到更早的前驱 ──→ 重新检查
  │       │
  │       └─→ 否 ──→ 检查:是否有足够执行时间?
  │               ├─→ 是 ──→ 选择为依赖窗口 ✓
  │               │
  │               └─→ 否 ──→ 前移到更早的前驱 ──→ 重新检查
  │
  └─→ 所有前驱都不合适 ──→ 报告"无安全窗口" ──→ 放弃该目标
```

### 依赖窗口选择标准

从目标算子向前找同一关键路径上的前序算子,优先选择:
- 离目标算子足够近(在同一 layer 或相邻 layer)
- 自身有一定执行时间(不是瞬时完成的小算子)
- 被 `npu-roofline-analysis` 判断为**非 memory-bound**
- 不是另一个大矩阵/大通信/大搬运算子

**重要**:即使前序算子本身是大型 MatMul(如 o_proj),其**输出**仍可作为依赖窗口,因为该算子执行完成后带宽空闲,可以开始预取下一个算子的权重。

### LongCat-Flash 实际使用模式

| 目标算子 | 依赖窗口 | 代码位置 | 说明 |
|---------|---------|---------|------|
| router.classifier | o_proj 输出 | modeling_longcat_flash.py:1244 | o_proj 完成后预取 router 权重 |
| gate_up_proj | o_proj 输出 | modeling_longcat_flash.py:210 | o_proj 完成后预取 MLP 权重 |
| down_proj | gate_up_proj 输入 x | modeling_longcat_flash.py:239 | 在 gate_up_proj 执行时预取 down_proj 权重 |
| q_a_proj(下一层) | down_proj 输出 | modeling_longcat_flash.py:1259 | down_proj 完成后预取下一层 attention 权重 |
| q_b_proj(下一层) | down_proj 输出 | modeling_longcat_flash.py:1261 | 同上 |
| kv_a_proj(下一层) | down_proj 输出 | modeling_longcat_flash.py:1263 | 同上 |
| experts.w2_weight | router_logits 输出 | ffn.py:76 | router 完成后预取 GMM2 权重 |
| router.classifier(下一层) | gmm2_out 输出 | ffn.py:137 | GMM2 完成后预取下一层 router 权重 |
| experts.w13_weight(下一层) | gmm2_out 输出 | ffn.py:139 | GMM2 完成后预取下一层 GMM1 权重 |

**关键观察**:
1. **大型 MatMul 的输出可作为依赖窗口**:o_proj, down_proj, gmm2_out 都是大型 MatMul,但其输出用于预取下一个算子
2. **跨层预取**:在当前层的 down_proj 完成时预取下一层的 attention 权重
3. **MoE 链式预取**:router → GMM1 → GMM2 → 下一层 router,形成预取链

### 何时放弃预取

如果所有合理前驱都表现为重搬运或重通信窗口,则明确说明"当前阶段没有安全窗口",不要为了"有改动"而硬插 prefetch。

---

## 代码实现指南

### 1. 开关传递路径

检查开关传递链路:命令行/配置文件 → ModelRunner → 模型类 `__init__` → `forward`/`decode`/`prefill`

**配置文件示例**(`config.yaml`):
```yaml
model_config:
  enable_prefetch: true  # 默认 false
  enable_multi_stream: 2  # prefetch 需要多流支持
```

**模型初始化示例**:
```python
class YourModel:
    def __init__(self, config):
        self.enable_prefetch = config.get("enable_prefetch", False)
        # prefetch 需要图模式 + 多流
        assert not self.enable_prefetch or config.get("enable_multi_stream", 0) > 0
```

### 2. Prefetch 调用模式

**推荐:使用 wrapper 函数**(`executor/utils/common_utils.py`):
```python
def npu_prefetch(switch_flag, weight, depend, size, offset=0):
    if switch_flag:
        return torch_npu.npu_prefetch(weight, depend, size, offset)
    else:
        return None
```

**在模型中调用**:
```python
from executor.utils.common_utils import npu_prefetch

class YourModel:
    def forward(self, x):
        # 计算依赖窗口
        dependency_output = self.some_layer(x)

        # 预取下一个算子的权重
        npu_prefetch(
            self.enable_prefetch,
            weight=self.next_layer.weight,
            depend=dependency_output,
            size=self.prefetch_size,
            offset=0
        )

        # 执行目标算子
        output = self.next_layer(dependency_output)
        return output
```

### 3. 图模式下的依赖节点

- 图模式下,`dependency` 必须是明确的 tensor 节点
- 依赖节点必须与目标算子在逻辑时序上匹配
- 不要跨分支、跨阶段乱复用依赖节点

**正确示例**:
```python
# 依赖节点是目标算子的直接前驱
attn_output = self.attention(x)  # 依赖窗口
npu_prefetch(self.enable_prefetch, self.mlp.weight, attn_output, size)
mlp_output = self.mlp(attn_output)  # 目标算子
```

**错误示例**:
```python
# 依赖节点与目标算子不在同一路径
attn_output = self.attention(x)
npu_prefetch(self.enable_prefetch, self.mlp.weight, x, size)  # 错误:x 不是 mlp 的直接前驱
mlp_output = self.mlp(attn_output)
```

### 4. 最小改动原则

优先:
- 单个目标算子
- 单个稳定窗口
- 单个保守 `max_size`
- 单文件或单模块的最小侵入式改动

避免:
- 多处同时预取
- 多个重量级目标一起上
- 全量权重一次性预取
- 改动多个并行策略和 prefetch 策略混在一个 patch 里

### 5. 仓库内的先验经验

| 场景 | 经验 |
|------|------|
| **LongCat-Flash** | 优先看 QBMM、MLAProlog、Matmul;对更重的后续算子,可提前预取更早权重 |
| **AFD/FFN 分离** | Attention 提速后 FFN 成瓶颈时,优先看 GMM/Matmul;仅在通信间隙和非重搬运窗口存在时尝试 |
| **MoE** | 专家计算常见候选是 grouped_matmul 和大专家线性层;dispatch/combine/router 通常不是 prefetch 目标 |
| **Dense MLP** | gate_up_proj/down_proj/o_proj 常是候选;但其前序窗口若已被另一个 MatMul 占满,不要硬加 |

---

## 性能测试验证

### 验证检查清单

在 prefetch 前后运行 profiling,按以下清单逐一检查:

- [ ] **目标算子执行时间是否下降**:对比 prefetch 前后目标算子的耗时
- [ ] **关键路径是否缩短**:对比主流上的总耗时或空洞/等待时间
- [ ] **依赖窗口是否稳定**:前序算子耗时不应明显增加
- [ ] **是否出现新的性能瓶颈**:检查其他流是否出现新的长拖尾
- [ ] **功能正确性是否保持**:对比 prefetch 前后输出一致性

### 成功标准

至少满足其中**两项**:
1. 目标算子耗时下降
2. 关键路径缩短
3. 前序窗口没有明显退化
4. 功能结果无差异

### 失败信号与调整策略

| 失败信号 | 根因分析 | 调整策略 |
|----------|---------|---------|
| 目标算子没快,前序窗口反而变慢 | `max_size` 过大,争抢带宽 | 减小 `max_size`:50% → 30% → 10% |
| 关键路径没有缩短,只是等待位置挪动 | 依赖窗口选择不当 | 前移或后移 prefetch 位置 |
| 多流上出现新的长拖尾 | prefetch 引入新的同步点 | 调整依赖节点或放弃该位置 |
| 图模式下依赖关系不合法 | 依赖节点与目标算子不匹配 | 修正依赖节点或改为 eager 模式测试 |

### 调优顺序

1. **先固定位置,只调 `max_size`**:从 50% → 70% → 100% 或 50% → 30% → 10%
2. **再固定 `max_size`,微调 dependency 位置**:前移或后移依赖窗口
3. **最后才考虑多目标叠加**:单个目标验证成功后再扩展

---

## 常见问题与修复

| 错误现象 | 根因 | 修复方案 |
|---------|------|---------|
| 不先跑 roofline 就选目标 | 缺少 profiling 数据支持 | 必须先用 `npu-roofline-analysis` 确认 memory-bound |
| 选了 LayerNorm/ROPE 作为预取目标 | 误判目标算子 | 这些通常是依赖窗口,不是预取目标 |
| dependency 窗口也是 memory-bound | 依赖窗口选择不当 | 选非 memory-bound 的前驱,或前移 prefetch |
| 一次性预取全量权重 | `max_size` 过大 | 先保守 `max_size`(50%),后续再调优 |
| 多处同时加 prefetch | 改动范围过大 | 先单个目标,验证后再扩展 |
| 没有开关保护 | 缺少回退机制 | 所有 prefetch 必须有 `enable_prefetch` 开关 |
| 前序窗口退化但不调整 | 未根据 profiling 调整 | 缩小 `max_size` 或调整位置,必要时回退 |
| 图模式下编译失败 | 依赖节点不合法 | 确保依赖节点与目标算子在同一路径 |

---

## 输出要求

完成后必须汇报:
1. **选了哪些目标算子,为什么**:基于 roofline 分析结果或用户手动指定
2. **roofline skill 的关键结论**:memory-bound 算子列表、关键路径(如使用自动化分析)
3. **为什么选这个 dependency 窗口**:非 memory-bound、有足够执行时间
4. **加了哪些开关与代码路径**:配置文件 → 模型初始化 → forward 调用
5. **`max_size` 取值依据**:权重维度、数据类型、保守系数或用户指定值
6. **做了哪些验证**:profiling 对比、功能验证
7. **还有哪些风险或后续调优建议**:潜在问题、下一步优化方向

**必须交付的文件**:
1. **修改后的代码文件**:包含 prefetch 实现的模型代码
2. **测试脚本**:用于对比 baseline 和优化后性能的自动化测试脚本
3. **性能优化报告**:Markdown 格式,包含:
   - 优化配置和参数
   - 代码修改说明
   - 性能测试结果(如有模型权重)
   - 验证检查清单
   - 后续调优建议
   - 如何运行测试的说明

**如果无法运行实际测试**(如缺少模型权重):
- 创建完整的测试脚本和报告模板
- 说明如何运行测试
- 提供预期的性能改善分析

---

## 参考资料索引

| 主题 | 文件路径 |
|-----|---------|
| npu_prefetch wrapper 函数 | `executor/utils/common_utils.py` |
| LongCat-Flash prefetch 实现 | `models/longcat-flash/models/modeling_longcat_flash.py` |
| FFN prefetch 大小计算 | `models/longcat-flash/models/ffn.py` |
| 配置验证逻辑 | `models/longcat-flash/models/model_setting.py` |
| 配置示例 | `models/longcat-flash/config/README.md` |
| Roofline 分析 | 调用 `npu-roofline-analysis` skill |
| 图模式适配 | 调用 `model-infer-graph-mode` skill |
