# VEC Bound 优化策略

VEC bound 是 elementwise、activation、reduction 类算子最常见的瓶颈。Vector 计算单元成为耗时主导，MTE2/MTE3 搬运单元相对空闲。

---

## 判定条件

- Vector 单元利用率高，`aiv_vec_ratio` 是耗时主导
- Vector 指令占据主要执行时间
- MTE2 和 Cube 单元相对空闲

**瓶颈严重程度分级**：

| VEC 占比 | 等级 | 优化方向 |
|----------|------|---------|
| 50–65% | 轻度 | DoubleBuffer + UB 融合有较大收益 |
| 65–80% | 中度 | 减少 Cast 或融合指令 |
| >80% | 深度 | VEC 本身已接近理论极限，优化空间有限 |

---

## 仿真图分析要点

- 识别 Vector 指令密集区
- 检查 Vector 指令间的数据依赖性，寻找可并行执行的机会

**VEC bound trace 特征**：

```
Time: 0 ---------------------------------------- 100ms

SCALAR     |##............................| 5%
SCALARLDST |##............................| 4%
MTE2       |.##....##....##....##.........|15%
VECTOR     |..############################|65%  <- 主导
MTE3       |..........................####.|10%
```

VECTOR 行持续活跃，MTE2 有明显空闲等待，说明搬运速度快于计算速度。

---

## VEC 内部指令分析

从 Chrome Trace JSON 提取 pid=5 (VECTOR) 事件，按 `name` 分类统计：

| 指令类型 | 典型指令 | 含义 |
|---------|---------|------|
| 算术类 | `vec_add`, `vec_mul`, `vec_sub` | 基本计算，低延迟 |
| 超越函数类 | `vec_exp`, `vec_log`, `vec_rec`, `vec_rsqrt` | 高延迟指令，无硬件加速 |
| 类型转换类 | `vcvt_f2f`, `vcvt_f2s`, `vcvt_s2f` | Cast 开销 |
| 归约类 | `vec_reduce_sum`, `vec_reduce_max` | reduction 开销 |

**判断规则**：
- Cast 占比 >20% → 类型转换密集场景，优化 Cast 有高收益
- 超越函数占比 >30% → 深度 VEC bound，优化空间有限

> ascend950 regbase 范式使用 **RVEC** 单元：RVECEX（执行）、RVECLD（加载）、RVECST（存储）、RVECSU（设置）。

---

## 策略 1：UB 融合

多个连续的 Vector 操作直接在 UB 中完成，不将中间结果写回 GM，消除不必要的 MTE2/MTE3 往返。

```
未融合: GM → UB → Compute1 → GM → UB → Compute2 → GM    // 6 次 GM 访问
已融合: GM → UB → Compute1 → Compute2 → UB → GM          // 2 次 GM 访问
```

**检查方法**：在 trace 中观察两个 VECTOR 活跃段之间是否插入了 MTE3（写回）+ MTE2（读入）。如果有，说明中间结果经过了 GM，未融合。

| 操作 | 说明 |
|------|------|
| 识别可融合的相邻 Vector 操作 | 消除中间搬移和暂存 |
| 链式 Vector 操作合并 | Mul+Add → MulAdd, 多个激活函数链式处理 |
| 减少中间结果写回 UB | 融合后在寄存器内完成传递 |

---

## 策略 2：减少类型转换

Cast（类型转换）是 VEC bound 中最常见的隐性开销。典型模式：`fp16 → Cast fp32 → 计算 → Cast fp16`。当计算本身只有 1–2 条指令时，Cast 可能占总 VEC 时间的 30–50%。

| 操作 | 说明 |
|------|------|
| 批量 Cast | 将多次 Cast 合并为一次大粒度操作 |
| 避免不必要的 Cast | 检查计算精度是否必须转换 |
| 选择合适的计算精度 | 全链路 fp16 或全链路 fp32，避免往返转换 |

---

## 策略 3：融合指令

使用融合指令减少 VEC 指令数：

| 指令 | 等价操作 | 说明 |
|------|---------|------|
| VMULA | VMUL + VADD | 乘加融合 |
| VMULS | VMUL + VSUB | 乘减融合 |
| VMADD | 累加模式 | 单指令累加 |

---

## 策略 4：低延迟归约

对于 reduction 操作，优先使用硬件树形归约指令（ReduceSum / ReduceMax / ReduceMin），避免手动 for 循环逐元素归约。

---

## 策略 5：RegBase 访存

| 操作 | 说明 |
|------|------|
| 数据布局调整为 RegBase 友好 | 连续读取、对齐访问 |
| 使用 RegBase 加载指令 | 减少地址计算开销 |
| 减少 vload/vstore 次数 | RegBase 大粒度搬移优势 |

---

## 策略 6：减少计算量（VEC BOUND 核心策略）

深度 VEC BOUND（>80%）时，微架构调整（DoubleBuffer、融合指令、Bank Align）收益有限。瓶颈在计算量本身，须从**算法层面**减少向量指令数。

### 适用场景

所有使用硬件超越函数（Exp、Log、Tanh、Sigmoid、GELU、Softmax）的 elementwise / activation 算子。这些函数内部使用 Taylor 级数展开，每个元素需数十条 FMA 指令，是计算量的主要来源。

### 优化手段

| 手段 | 说明 | 适用条件 |
|------|------|---------|
| **LUT + 插值** | 用查表+插值替代 Taylor 展开，减少计算量 | 从 asc-devkit 查对应 API（如 `Exp<T, expandLevel, isReuseSrc>` 设 `expandLevel=0` 即为查表模式）；`Log`、`Tanh` 等超越函数同理 |

### 案例分析：Exp 算子

`AscendC::Exp<T, expandLevel, isReuseSrc>` 的 `expandLevel = 0` 即内部走查表（LUT）+ 插值，无需手写查表逻辑。优先用此模式，减少 Taylor 项数（12→8）有精度风险，不推荐。

### 检查方法

在 profiling 数据中检查 `aiv_vec_ratio`：
- >90% 且算子是 elementwise/activation 类 → 优先考虑减少计算量
- 从 Chrome Trace 识别 `vec_exp`/`vec_log`/`vec_tanh` 等超越函数指令占比

---

## DoubleBuffer 检查

在 Chrome Trace 中观察 MTE2 和 VECTOR 行的时间重叠：

| 模式 | 特征 | 含义 |
|------|------|------|
| **DB 生效** | MTE2 和 VECTOR 交替出现 | 搬入与计算重叠，流水健康 |
| **DB 未生效** | MTE2 全在前，VECTOR 全在后 | 串行执行，需开启或修复 DoubleBuffer |

---

## Tiling 修正建议

- 调整 UB 布局以支持更高效的 RegBase 访问模式
- 调整 tile 粒度匹配 Vector 融合窗口
- 增大 tile 尺寸减少循环次数，降低流水启停开销（UB 容量：192KB / 910B2，248KB / 950）
- 使能 DoubleBuffer 时，实际可用 UB 需除以 2
