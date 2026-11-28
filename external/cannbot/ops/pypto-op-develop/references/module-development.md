# 逐模块实现

读取 DESIGN.md 和模块接口，按确定的模块划分实现。模块用于组织开发和验证，不要求每个模块都有独立的 JIT kernel。

## 单模块与多模块

单模块直接产出 `<op>_impl.py`，使用完整 golden 验证。
多模块每次实现一个模块，验证通过后再实现下一个。验证时可以导出中间结果，或使用 golden 提供的输入张量，但被验证的计算必须与最终实现一致。

当前编排使用累计实现文件，如 `<op>_module1_impl.py`、`<op>_module12_impl.py`；命名及调度约定见[编排规则](../../pypto-orchestration-manual/references/rules.md)。

## 实现准备

核对 API 签名、输入输出 shape/dtype、切片秩和偏移、尾块有效范围、循环状态与 tile 配置。
采用 DESIGN.md 中确定的 Tile、显式配置和类型转换位置；其中的候选值仅供后续比较，不表示已验证可用。关键值未确定时先补全设计。调整参数时遵守记录的限制，并重新验证受影响结果。
使用[实现模板](../templates/impl_template.py.tmpl)和[执行约束](execution-constraints.md)；遇到具体问题再读[调试参考](../../pypto-general-debug/references/DEBUG_GUIDEBOOK.md)。

实现应覆盖 golden 中属于本模块的每个计算，包括掩码、缩放、cast、状态更新和全部输出。逐项对照计算清单，补全遗漏后再请求验证。

## 验证与失败定位

将实现文件路径和模块输出交给验证 Agent，检查编译、shape/dtype，并用 `detailed_tensor_compare` 与 golden 比较数值。记录实际结果；编译或结构检查不能代替精度比较。
DESIGN.md 的设计检查结论只说明已核对的文档、推导或接口，不能直接作为实现测试结果；按其验证方案执行并记录实际证据。

验证通过后记录模块输出的比较结果，继续下一个模块。后续失败时先找出最早输出错误的模块，检查其输入输出及中间值，必要时二分定位。不要因后续失败随意修改已验证模块；确需修改时重新验证该模块及依赖它的结果。

已知问题可参考[通用调试](../../pypto-general-debug/SKILL.md)，精度问题分别参考[精度调试](../../pypto-precision-debug/SKILL.md)和[中间值比较](../../pypto-precision-compare/SKILL.md)。每个调整必须对应具体原因，不能盲目叠加精度变通参数。

全部模块完成后验证组合及所有最终输出，确认临时输入或检查点没有进入生产计算路径。

## Kernel 配置与计算函数

JIT 入口使用 `@pypto.frontend.jit`。计算较复杂或需要复用时，可把 PyPTO 计算放入普通 helper，由 JIT 入口调用；helper 仍在构图环境中执行，不能把它当作独立的 CPU 数值函数。

```python
def compute(x, output):
    # 根据设计完成切分、加载、计算和写回。
    ...

@pypto.frontend.jit
def kernel(x, output):
    compute(x, output)
```

根据具体实现选择 `pass_options` 和 `runtime_options`，不照搬固定取值：

| 参数 | 用途 |
|---|---|
| `cube_l1_reuse_setting` | 配置 Cube L1 复用 |
| `vec_nbuffer_setting`、`cube_nbuffer_setting` | 配置缓冲副本数，需计入资源预算 |
| `stitch_function_max_num` | 配置子图合并数量 |
| `device_sched_mode` | 选择目标设备支持的调度方式 |

需要识别不同计算段时，可用 `pypto.set_semantic_label("scores")` 等有含义的名称；标签用于定位代码段，不替代正确的数据依赖和 Tile 配置。

## 循环与 Tile

循环选择和返回值见[循环约束](../../pypto-op-design/constraints/loop.md)，Tile 的秩、对齐及配置时机见[Tiling 约束](../../pypto-op-design/constraints/tiling.md)。
需要按展开因子调整每次处理的数据量时，使用 `loop_unroll` 返回的因子；多个展开候选会增加编译路径。

## 数据读取与写回

| 场景 | 表达方式 | 注意事项 |
|---|---|---|
| 连续 tile | `pypto.view(tensor, shape, offsets, valid_shape=...)` | 显式指定有效范围时使用 view |
| 简单连续切片 | `tensor[start:stop, :]` | 核对目标版本的索引及有效形状支持 |
| 分页 KV | view、索引及结果拼装 | 参考[分页加载模式](../../pypto-op-design/patterns/atoms/AT-17-block-gather.md) |
| 稀疏索引读取 | `pypto.index_select` | 核对索引轴、类型和边界 |
| 按模式选择元素 | `pypto.gathermask` | 例如 RoPE 奇偶位置，核对 mode 的含义 |
| 输出 tile | `pypto.assemble(tile, offsets, output)` 或切片赋值 | 核对秩、偏移和输出有效区域 |
| Cache 更新 | `pypto.scatter_update` | 明确索引和重复写入的行为 |

## 动态长度与状态位置

```python
length = tensor.shape[axis]
tile_count = (length + tile_size - 1) // tile_size
valid_len = (length - offset).min(tile_size)

# 需要将复杂符号表达式标记为中间变量时，单独调用：
seq_len = cu_seqlens[batch + 1] - cu_seqlens[batch]
seq_len.as_variable()
```

`as_variable()` 原地修改符号对象、返回 None。它不负责分配 Tensor，也不自动建立循环状态。

| 状态 | 典型位置 | 原因 |
|---|---|---|
| Online Softmax 累积器 | Q tile 循环内、KV tile 循环外 | 每个 Q tile 独立累积 |
| 递推状态 | Batch 循环内、序列循环外 | 每个序列保持独立状态 |
| 单轮临时张量 | 当前循环体内 | 无跨迭代依赖 |

需要全零状态时使用目标 API 支持的分配和填充方式，例如 `pypto.full`；在正确作用域初始化，避免每轮重置。

## 数据类型与精度

在具体计算中标明输入、累加、转换和输出类型。常见的浮点归约采用 FP32 累加后转回输出类型；INT8 矩阵乘可产生 INT32 累加结果再反量化。FP8 的缩放及转换位置取决于所选 API，不能套用统一路径。

除法可按目标 API 选择 `pypto.PrecisionType.HIGH_PRECISION` 或 `INTRINSIC`，并与 golden 比较误差。规则见[API 约束](../../pypto-op-design/constraints/api.md)，量化计算见对应 AT 卡片。
