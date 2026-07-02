# spec.yaml 范例

8 个范例覆盖典型复杂度阶梯。**全部通过 9 stage 校验**（stage 9 因测试环境未装 torch 走 SKIP，不算失败）。可以直接拷出来作为新算子起手模板。

| 算子 | 复杂度 | category | 主要 paradigms | stage 通过情况 | 教学点 |
|---|---|---|---|---|---|
| [add](add/spec.yaml) | 入门 | elementwise | Elementwise + Broadcast | ✅ 1-8 PASS · ↷ 9 SKIP | 双输入 numpy promote、algebraic invariant（交换律 / 加法零元）、`patterns[]` 数组写法 |
| [softmax](softmax/spec.yaml) | 中等 | reduction_composite | Reduction + NumericalStable + FusedComposite | ✅ 1-8 PASS · ↷ 9 SKIP | dim attribute 自动注入、accumulator_dtype 必填、composition 五原语链（max-shift + exp + reduce + div）、4 种 invariant 综合（elementwise_ge/le + reduce_equals + no_leak_intermediates）、AP-004 反模式联动 |
| [cumsum](cumsum/spec.yaml) | 中等 | recurrence | Recurrence | ✅ 1-8 PASS · ↷ 9 SKIP | 沿轴累加、accumulation_order: sequential、dim attribute |
| [dropout](dropout/spec.yaml) | 中等 | random_sampling | Elementwise + RandomSampling | ✅ 1-8 PASS · ↷ 9 SKIP | seed attribute 自动注入、training/eval 双路径、固定 seed 下 deterministic invariant |
| [matmul](matmul/spec.yaml) | 进阶 | contraction | Contraction + Broadcast | ✅ 1-8 PASS · ↷ 9 SKIP | 折叠维 + 显式维混合（`"...batch_a", "M", "K"`）、shape_rule 用 numpy_expr + IfExp 表达 transpose、explicit broadcast.rules、algebraic invariant（`equals_when_input_is_zero` / `associative_along_batch`） |
| [complex](complex/spec.yaml) | 进阶 | layout_transform | Elementwise + Broadcast + LayoutTransform | ✅ 1-8 PASS · ↷ 9 SKIP | 双 input 不同 dtype（real/imag 各 fp32/fp64）、IfExp dtype_rule（fp→complex64/128）、numpy promote 行 + complex 输出 |
| [nonzero](nonzero/spec.yaml) | 进阶 | variable_output | IndexGather + VariableOutput | ✅ 1-8 PASS · ↷ 9 SKIP | data_dependent shape_rule_kind、shape_bounds.max_elements 上界声明、static_dims/dynamic_dims 拆分、固定输出 dtype int64 |
| [fused_quant_matmul](fused_quant_matmul/spec.yaml) | 高阶 | fused_composite | Contraction + Quantization + Elementwise + FusedComposite | ✅ 1-8 PASS · ↷ 9 SKIP | composition DAG oracle（5 节点链）+ Quantization + FusedComposite 三合一；CANN 公式 2（FP bias）+ gelu_tanh |

> 安装 torch 后 stage 9 也会变 ✅（`pip install torch`）。

## 阅读顺序

1. **add** — 先看一眼，理解 16 个顶层字段长什么样
2. **softmax** — 覆盖范式最多，最佳代表（数学性质机器化 + composition 白盒分解）
3. **cumsum** / **dropout** — Recurrence、RandomSampling 范式骨架
4. **matmul** — contraction 类的 shape 表达（folded + IfExp + explicit broadcast.rules）
5. **complex** — 多 input 不同 dtype + IfExp dtype_rule
6. **nonzero** — data_dependent shape_rule 的范式（输出形状由输入值决定）
7. **fused_quant_matmul** — FusedComposite 的真实落地（量化 + matmul + bias + 激活的 composition DAG）

## 跑校验

```
# 单个
python3 ../scripts/validate_spec.py add/spec.yaml

# 全部（用 glob 兜底，新增 example 不需要改命令）
for d in */; do
  python3 ../scripts/validate_spec.py "${d}spec.yaml"
done

# pytest 回归
pytest ../tests/test_examples.py -v
```

## 怎么用作模板

1. 选与自己算子最接近的范例
2. 复制 `spec.yaml` 到 `ops/<your_op>/spec.yaml`
3. 改 `op.name` / `op.description` / `op.category` / `op.paradigms`
4. 改 `inputs` / `outputs` / `attributes`
5. 重写 `math_semantics.formula` 与 `reference_oracle.api`
6. 调整 `boundary_conditions` / `extreme_inputs`
7. 跑 `validate_spec.py` 直到 PASS

## 不在范例里的范式

8 个范例覆盖 elementwise / reduction_composite / contraction / fused_composite / layout_transform / recurrence / random_sampling / variable_output 八类。其他 17 类算子起手时，先用**生成器**搭骨架：

```
python3 ../scripts/generate_spec.py \
    --op-name <your_op> --category <category> --paradigms <PascalCase,...> \
    --inputs "<name>:<dtype1>,<dtype2>" --outputs <name> \
    --output-dir <path>
```

生成的骨架会按 paradigms 自动注入对应字段（accumulator_dtype / composition 占位 / dim attribute 等）。
