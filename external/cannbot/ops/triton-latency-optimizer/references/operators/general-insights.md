# Triton-Ascend 通用优化洞察

> 已剔除与 `latency-optimizer` 重复的内容（避免标量降级、循环不变量外提、Load/Store 重排序通用原理、Autotune 通用规则等）。

---

## 1. 先判断算子瓶颈类型

在动手优化之前，先判断算子属于哪一类瓶颈：

- **计算瓶颈**：kernel 内大量浮点/矩阵运算，ALU 或 Cube 是瓶颈。
- **内存瓶颈**：每个线程加载/存储大量数据，实现延迟远大于框架延迟。

**经验**：对于内存瓶颈算子，优先减少全局内存访问量、优化访问模式、消除冗余读写；单纯优化算术指令收益很小。

---

## 2. 减少全局内存访问量

- 避免重复的 `tl.load`：如果一个值在多个地方用到，只加载一次。
- 避免冗余的 `tl.store`：当 `NEED_OUTPUT=False` 时，不要向输出缓存写入不必要的数据。
- **合并相邻的读写**：如果先 `store` 再 `load` 同一块数据，尝试用寄存器直接传递，避免二次全局内存访问。

**典型案例**：将 `v_base = ((b*N+n)*S+s)*R`、`k_base = ((b*N+n)*S+s)*P` 替换为已计算好的 `kv_base - cs_base` 和 `cs_base`，既减少重复计算，也帮助编译器复用地址生成逻辑。

> 更系统的 Pass 合并方法请参考 `latency-optimizer` 的 `references/pass-merge.md`。

---

## 3. 调整 Load / Store 顺序

- 将同一轮迭代中所有不依赖的 `tl.load` 提前到计算之前，可让内存访问与后续计算重叠。
- 将 `tl.store` 尽可能延后，避免下一次迭代的 `tl.load` 被上一次 `tl.store` 阻塞。

**注意**：只有在 load/store 之间没有真实数据依赖时才可重排，否则结果会错。

> 通用 Load 重排序规则请参考 `latency-optimizer` 的 `references/load-order.md`。

---

## 4. 循环不变量外提

凡是只依赖外层循环变量、在内层循环中不变的量，都应移到外层：

- 权重/参数向量（如 `gamma`）
- 分块参数（如 `dk0`、`dk1`、`dv0`、`dv1`、`d0_offs`）
- 一次性读取的索引（如 PA_BLK 模式中的 `idx_val`）

外提后减少了每轮内层循环的冗余 load/计算。

> 详细规则请参考 `latency-optimizer` 的 `references/loop-invariant-hoisting.md`。

---

## 5. 避免标量降级

Ascend Vector 单元偏好向量操作，以下写法容易退化成标量循环：

| 不推荐 | 推荐 |
|--------|------|
| `a % b`（int） | `a - (a // b) * b` |
| `tl.sqrt(x) / y` | `x * tl.rsqrt(y)`（在精度允许时） |
| 向量上的 `int64` 比较/除法 | 转成 `float32` 或 `int32` 向量比较 |
| Python 标量广播 | 转成等长向量常量 |

**注意**：`int32` 转换并非总是正收益。如果转换本身引入额外的 `.to(tl.int32)` 指令，而 kernel 又是内存瓶颈，反而可能让延迟变差。需要实测验证。

> 完整降级条件表请参考 `latency-optimizer` 的 `references/avoid_scalar_lowering.md`。

---

## 6. 用代数化简减少指令

- 利用 `H = R + P` 的关系：`v_base = kv_base - cs_base`。
- 利用已知常量合并乘除：例如 `idx % block_size` 在已知 `bn_id = idx // block_size` 时，可直接写成 `idx - bn_id * block_size`。

这类化简不仅减少指令数，也更容易被编译器识别为地址复用。

---

## 7. Autotune 要谨慎

Autotune 只能调那些**不影响输出语义和内存布局**的参数：

- 可以调：行/块分块大小、每个 program 处理的行数、warps/threads 配置。
- **不可以调**：决定输出张量物理布局的参数（如 NZ 缓存模式中的 `BLOCK_D0`）。改变它会导致 cache 排布变化，测试用例会大面积失败。

**建议**：给 autotune 加配置前，先确认该参数是否只影响调度/并行度，而不影响写入位置。

> 完整 autotune 指南请参考 `latency-optimizer` 的 `references/autotune.md`。

---

## 8. 编译器选项需实测

- `multibuffer=True` 在多数内存密集型算子上有收益，建议默认开启。
- `unit_flag` 等选项并非所有 Triton-Ascend 运行时都支持；如果遇到 "Please DO NOT tune args" 警告，立即回退。

---

## 9. 评估指标选择

- `speedup_vs_torch` 容易受框架延迟抖动影响，尤其当框架延迟本身很小（<0.1ms）时，几次测试的波动就能让比值变化几个点。
- 对于内存瓶颈算子，**implementation 平均延迟**是更稳定的优化指标。
- 多 shape 场景以几何平均加速比为准，避免被个别异常 shape 带偏。

---

## 10. 何时停止优化

当出现以下情况时，通常可以进入终局：

- `latency-optimizer` 技能清单中已无命中条件成立的优化点。
- 每次新优化带来的实现延迟提升 < 1%，且伴随编译/验证风险上升。
- 进一步优需要改变数据布局或引入复杂临时缓存，会触及正确性边界。

**原则**：在精度正确的前提下追求性能；当边际收益趋近于零时，保留已验证的最佳版本。

---

## 11. 验证与基线管理

- 每次优化后必须跑完整 `verify.py`，读取 `verify_result.json` 的 `passed_cases == total_cases`。
- Phase 4 与 Phase 3 基线对比时，使用几何平均 `speedup_vs_torch`；基线 verify 结果直接复用，不必重跑。
- 失败时先分类（A/B/C），A 类可修复后重试，B/C 类及时止损。
