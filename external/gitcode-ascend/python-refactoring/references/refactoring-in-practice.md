# Skill: Python 代码重构实战经验

## 触发条件

当用户要求"重构这个模块"、"参考历史重构经验"、"大规模重构"，或需要对多文件/多模块进行系统性重构时触发。本技能基于 vllm-ascend 仓库中真实的重构 PR 提炼而成。

## 目标

提供经过生产验证的重构模式和策略，帮助在大型 Python 项目中安全、高效地执行重构。

## 核心原则

- 渐进式重构优于一次性重写
- 减少与上游的分歧是下游项目重构的首要目标
- 每个 PR 只做一类重构，便于 review 和回滚
- 删除代码是最好的重构

---

## 1. 渐进式多 PR 重构策略

### 实战案例：Quantization 模块重构（4 个 PR）

vllm-ascend 的量化模块经历了一次经典的渐进式重构，分 4 个 PR 完成：

| 阶段 | PR | 做了什么 | 净删除行数 |
|------|-----|---------|-----------|
| 1/N | #2680 | 用字典映射替代 Quantizer 类层次 | -233 |
| 2/N | #2785 | 删除步骤 1 后暴露的死代码 | -454 |
| 3/N | #3021 | 将模型中的量化配置集中到 quant_config.py | -4 |
| 4/N | #5738 | 引入注册表 + ABC + 子包结构，完成架构升级 | +520 |

**关键模式：**

```
简化分发 → 删除死代码 → 集中职责 → 架构升级
```

**适用 skill 映射：**
- 阶段 1 → [code-smells.md] 过度继承、上帝类
- 阶段 2 → [code-smells.md] 死代码
- 阶段 3 → [design-patterns.md] 职责集中（减少上游分歧）
- 阶段 4 → [design-patterns.md] 工厂模式（注册表）、模板方法（ABC）

**实施要点：**
- 每个 PR 独立可合入、独立可回滚
- 前 3 步净删除代码，降低复杂度后再做架构升级
- 最终的架构升级 PR 虽然行数增加，但引入了清晰的抽象层

### 实战案例：MoE 模块重构（10+ 个 PR）

MoE 模块的重构跨越了更长的时间线，展示了一个模块从单文件到包结构的完整演进：

```
提取关注点 → 引入策略抽象 → 合并冗余文件 → 继承去重
→ 重命名/重组织 → 删除死路径 → 对齐上游 → 类型形式化
```

| 阶段 | 代表 PR | 模式 | 效果 |
|------|---------|------|------|
| 提取 | #2487, #2503 | 分离关注点 | all2allv 和 torchair 从单体文件中抽出 |
| 抽象 | #2570 | 策略模式 | token_dispatcher 替代多个 fused_experts_with_xxx |
| 合并 | #3176 | 文件合并 + 死代码删除 | 删除 935 行，两文件合一 |
| 去重 | #3365 | 继承去重 | MC2 继承 All2All，消除重复通信代码 |
| 重组织 | #3646 | 重命名 + 包化 | ops/moe → ops/fused_moe/ 包结构 |
| 清理 | #4224 | 删除劣势路径 | 基准测试证明 multicast 不如 all_gather，删除 |
| 对齐 | #5189 | 复用上游实现 | 删除自定义 all_reduce，复用 vLLM 逻辑 |
| 类型化 | #5481 | dict/tuple → dataclass | 提升类型安全和 IDE 支持 |

---

## 2. 七大实战重构模式

### 模式一：用数据结构替代类层次

**场景：** 类层次仅用于分发/选择，没有真正的多态行为。

**案例：** Quantization #2680 — AscendQuantizer/LLMQuantizer 类被字典映射替代。

```python
# Before: 类层次分发
class AscendQuantizer:
    def get_method(self, config):
        if config.type == "w8a8":
            return W8A8Method()
        elif config.type == "w4a16":
            return W4A16Method()
        ...

# After: 字典映射
QUANT_METHODS: dict[str, type[QuantMethod]] = {
    "w8a8": W8A8Method,
    "w4a16": W4A16Method,
}

def get_quant_method(config: QuantConfig) -> QuantMethod:
    cls = QUANT_METHODS.get(config.type)
    if cls is None:
        raise ValueError(f"Unsupported: {config.type}")
    return cls()
```

**适用 skill：** [code-smells.md] 过度继承 → [design-patterns.md] 工厂模式（注册表）

**判断标准：** 如果类的子类之间只有数据不同而没有行为差异，用字典替代。

---

### 模式二：分离关注点（单体文件拆分）

**场景：** 单个文件承担多个不相关职责，超过 500 行。

**案例：** MoE #2503 — torchair（图模式）代码从 fused_moe.py（eager 模式）中分离。

**实施步骤：**
1. 识别文件中的独立关注点（如 eager vs graph、通信 vs 计算）
2. 按关注点提取到独立文件
3. 更新所有导入路径
4. 确保原文件的公共 API 不变（可通过 re-export）

```python
# 分离前：fused_moe.py 同时包含 eager 和 torchair 逻辑
# fused_moe.py (800+ lines)
class EagerFusedMoE: ...
class TorchairFusedMoE: ...

# 分离后：
# fused_moe.py (400 lines) — eager 模式
class EagerFusedMoE: ...

# torchair_fused_moe.py (400 lines) — 图模式
class TorchairFusedMoE: ...
```

**适用 skill：** [code-smells.md] 上帝类、过大模块

---

### 模式三：策略模式替代函数变体

**场景：** 存在多个 `do_something_with_xxx` 函数，逻辑骨架相同但细节不同。

**案例：** MoE #2570 — 多个 `fused_experts_with_xxx` 被 `token_dispatcher` 策略替代，净删除 573 行。

```python
# Before: 函数变体
def fused_experts_with_all2all(hidden, weights, ...): ...
def fused_experts_with_mc2(hidden, weights, ...): ...
def fused_experts_with_allgather(hidden, weights, ...): ...

# After: 策略抽象
class TokenDispatcher(Protocol):
    def dispatch(self, hidden: Tensor) -> DispatchResult: ...
    def combine(self, expert_out: Tensor) -> Tensor: ...

class All2AllDispatcher(TokenDispatcher): ...
class MC2Dispatcher(TokenDispatcher): ...
class AllGatherDispatcher(TokenDispatcher): ...
```

**适用 skill：** [code-smells.md] 重复代码 → [design-patterns.md] 策略模式

**判断标准：** 3 个以上函数变体共享 >50% 的逻辑骨架时，引入策略模式。

---

### 模式四：对齐上游 / 复用上游实现

**场景：** 下游项目（fork、插件、扩展）自定义了上游已有的功能，造成维护负担和合并冲突。这是所有 fork 项目最重要的重构模式。

**案例（vllm-ascend）：**
- MoE #5189 — 删除自定义 all_reduce，复用 vLLM 逻辑（-38 行）
- Attention #5916 — 修复继承链，正确调用 `super().__init__()`
- Ops #6523 — 删除自定义 rotary_embedding C++ 算子，改用 torch_npu 原生实现（-1333 行）
- Model Runner #6043 — 拆分方法对齐 GPUModelRunner 的结构

**通用判断标准：**

| 情况 | 操作 |
|------|------|
| 上游已有等价实现 | 直接删除自定义代码，调用上游 |
| 上游有基类/接口 | 继承并只覆盖差异部分 |
| 上游有约定的方法结构 | 对齐方法拆分和命名，降低认知差异 |
| 上游有配置/注册机制 | 通过配置扩展而非 monkey-patch |
| 上游即将支持你的需求 | 等待上游合入，提前适配接口 |

```python
# Before: 自定义实现，重复了基类逻辑
class AscendMLAMetadataBuilder:
    def __init__(self):
        # 重复了基类的初始化逻辑
        self.workspace_size = 128 * 1024
        self.chunk_size = ...
        ...

# After: 继承上游基类，仅覆盖差异
class AscendMLAMetadataBuilder(MLACommonMetadataBuilder):
    def __init__(self, ...):
        super().__init__(...)  # 复用基类逻辑
        self.workspace_size = 128 * 1024  # 仅覆盖差异部分
```

**对齐上游的收益量化：**
- 减少合并冲突：上游更新时无需手动同步自定义实现
- 减少测试负担：上游已测试的逻辑无需重复测试
- 减少代码量：Ops #6523 单次删除 1333 行

**适用 skill：** [code-smells.md] 重复代码 → [readability.md] 结构改进

---

### 模式五：职责归位（Move Method to Owning Class）

**场景：** 操作某个对象数据的逻辑散落在外部类中。

**案例：** #6041 — full-graph 参数更新逻辑从 acl_graph.py（358 行删除）移入各 Backend 类。

```python
# Before: 外部类管理所有 backend 的参数更新
class ACLGraph:
    def update_params(self, backend_type, ...):
        if backend_type == "attention":
            # 50 行 attention 参数更新逻辑
        elif backend_type == "mla":
            # 50 行 MLA 参数更新逻辑
        ...

# After: 每个 backend 管理自己的参数
class AscendAttentionBackend:
    def update_full_graph_params(self, ...):
        # attention 参数更新逻辑

class AscendMLABackend:
    def update_full_graph_params(self, ...):
        # MLA 参数更新逻辑

# 统一入口
def update_full_graph_params(backend, ...):
    backend.update_full_graph_params(...)
```

**适用 skill：** [code-smells.md] 上帝类 → [design-patterns.md] 模板方法 / 多态

---

### 模式六：基准驱动的死路径删除

**场景：** 代码中存在多条执行路径，但部分路径已被证明劣于其他路径。

**案例：** MoE #4224 — multicast 通信策略在双系统背靠背组网中仅 3 tps，而 all_gather 达 10 tps，删除 multicast 路径（-249 行）。

**实施要点：**
- 必须有基准测试数据支撑删除决策
- 在 PR 描述中记录性能对比数据
- 删除时同步清理配置项、测试、文档中的相关引用
- 不要保留注释掉的代码或 `# removed: multicast` 标记

**适用 skill：** [code-smells.md] 死代码

---

### 模式七：类型形式化（dict/tuple → dataclass）

**场景：** 函数间通过 dict 或 tuple 传递复合数据，字段含义不明确。

**案例：** MoE #5481 — MoECommMethod 和 MoETokenDispatcher 的输出从 dict/tuple 改为 dataclass（-57 行净减少）。

```python
# Before: 字典传递，字段含义靠注释
def dispatch(self, hidden) -> dict:
    return {
        "dispatched_input": dispatched,
        "tokens_per_expert": counts,
        "recv_counts": recv,
        ...
    }

# After: dataclass，IDE 可跳转、可补全
@dataclass
class DispatchResult:
    dispatched_input: torch.Tensor
    tokens_per_expert: torch.Tensor
    recv_counts: list[int]

def dispatch(self, hidden) -> DispatchResult:
    return DispatchResult(
        dispatched_input=dispatched,
        tokens_per_expert=counts,
        recv_counts=recv,
    )
```

**适用 skill：** [code-smells.md] 过长参数列表 → [readability.md] 类型标注

**判断标准：** dict/tuple 有 3 个以上字段，或在 2 个以上函数间传递时，改用 dataclass。

---

## 3. 重构安全守则

### 从真实回滚中学到的教训

MoE 的 "Remove manual memory cleanup" (#3365) 在合入后被 revert (#3483)，随后又被 reapply (#3512)。这说明：

1. **大型重构必须有充分的测试覆盖**
   - 单元测试覆盖核心逻辑
   - 集成测试覆盖端到端路径
   - 性能测试确认无回归

2. **每个 PR 保持原子性**
   - 一个 PR 只做一类重构
   - 回滚一个 PR 不应影响其他重构成果
   - PR 之间可以有依赖，但每个 PR 独立可工作

3. **重构 PR 的 commit message 规范**
   ```
   [Refactor] 简短描述做了什么

   为什么要重构：
   - 当前问题描述

   做了什么：
   1. 具体变更 1
   2. 具体变更 2

   验证方式：
   - 通过了哪些测试
   - 性能对比数据（如适用）
   ```

---

## 4. 重构规模评估矩阵

在开始重构前，用此矩阵评估规模和风险：

| 维度 | 小型 | 中型 | 大型 |
|------|------|------|------|
| 影响文件数 | 1-3 | 4-10 | 10+ |
| 净代码变更 | <100 行 | 100-500 行 | 500+ 行 |
| 是否改变公共 API | 否 | 部分 | 是 |
| 建议 PR 数 | 1 | 1-2 | 3+（渐进式） |
| 是否需要 RFC | 否 | 视情况 | 是 |
| 回滚风险 | 低 | 中 | 高 |

**大型重构必须：**
- 先写 RFC 或设计文档
- 拆分为多个 PR，每个 PR 独立可合入
- 前几个 PR 以删除和简化为主，最后再做架构升级
- 每个 PR 合入后观察 CI 和线上表现再继续下一个

---

## 执行流程

1. 评估重构规模（使用上方矩阵）
2. 如果是大型重构，制定分阶段计划
3. 每个阶段选择合适的重构模式（参考七大模式）
4. 映射到具体的 skill（code-smells / design-patterns / readability）
5. 实施重构，确保每个 PR 原子性
6. 验证：测试通过 + 性能无回归 + 公共 API 兼容
