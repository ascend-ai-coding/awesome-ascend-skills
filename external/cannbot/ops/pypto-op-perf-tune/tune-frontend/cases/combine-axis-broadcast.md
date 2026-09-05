# 案例：尾轴 Broadcast 合轴优化（combine_axis）

## 场景

Online softmax 更新阶段存在 `[4, 128] * [4, 1]` 和 `[4, 128] / [4, 1]` 等尾轴为 1 的 broadcast vector 计算。默认情况下，编译器先将 `[4, 1]` broadcast 到 `[4, 128]` 再做双目运算，产生额外的数据搬运。

## 优化方法

在 JIT 函数体首行添加 `pypto.experimental.set_operation_options(combine_axis=True)`，让编译器在代码生成阶段将尾轴 broadcast 内联为 brcb 指令，无需先展开再计算。

```python
@pypto.frontend.jit(...)
def kernel(...):
    pypto.experimental.set_operation_options(combine_axis=True)
    # ... kernel body
```

## 诊断方法

### 1. 静态分析：是否存在尾轴为 1 的 binary 操作

扫描算子的 tensor shape，查找所有形如 `[M, 1]` 的 tensor 参与的二元运算：

| tensor | shape | 来源 | 参与运算 |
|--------|-------|------|---------|
| `sum_update` | `[g_tile, 1]` | `pypto.sum(keepdim=True)` | `out * update_mul`（`[4,128]*[4,1]`） |
| `max_update` | `[g_tile, 1]` | `pypto.amax(keepdim=True)` | `out / sum`（`[4,128]/[4,1]`） |
| `update_mul` | `[g_tile, 1]` | `exp(sub)` 结果 | 同上 |
| `sum_local` | `[g_tile, 1]` | `pypto.sum(keepdim=True)` | `sum * scale + sum_local` |

**判断标准**：满足以下条件可尝试 `combine_axis=True`：
1. 存在形如 `[M, 1] * [M, N]` 的 broadcast binary 操作
2. 尾轴为 1 的 tensor 由前序 reduce 操作（`sum`/`amax` 等 + `keepdim=True`）产生，保证 GM 中连续
3. 该操作在内层热循环中频繁执行

### 2. 编译日志验证

开启 `compile_debug_mode=1` 后检查编译日志中是否出现 `brcb` 相关指令。若出现则说明 combine_axis 已生效。

## 约束条件

- 尾轴 broadcast 输入尾轴**必须连续**，否则功能失效
- `pypto.sum(keepdim=True)` 和 `pypto.amax(keepdim=True)` 的输出能保证在 GM 连续，符合条件
- 若前序是 COPY_IN，需在前端保证在 GM 连续
- 设置是**局部**的，只影响当前 jit/loop 作用域内的编译过程

## 效果分析

| 配置 | Pangu-7B Fused Layer（avg 10 iter） |
|------|-----------------------------------|
| 无 combine_axis | 459.38 us |
| `combine_axis=True` | 460.50 us |
| 差异 | **+0.24%（噪声范围内）** |

### 为什么收益不明显？

1. **Cube 占主导**：该算子主要耗时在 QKV/O/Gate/Up/Down 共 5 次 matmul（cube 计算），vector 操作占比低
2. **Broadcast 操作次数少**：尾轴 1 的 broadcast 只在内层 `LOOP_s2` 中出现 2 次（`out*update_mul` 和 `out/sum`），外层 `LOOP_n2` 共 8 次迭代，总次数有限
3. **Online softmax 的 vector 操作本身体量小**：`[4,128]` 和 `[4,1]` 的数据量仅 ~2KB，搬运开销可忽略

## 适用场景

`combine_axis=True` 在以下场景预期有收益：

- Vector 密集的算子（如纯 softmax、layer norm、激活函数）
- 尾轴 1 的 broadcast 操作处于最内层热循环，且执行次数多
- 非 Cube 密集型算子，vector 操作占比高

## 关键经验

1. **先诊断，再优化**：通过静态分析确认存在尾轴 1 broadcast 后再尝试，不要盲目添加
2. **关注热路径**：只在最内层循环或高频执行路径上的 broadcast 值得优化
3. **适合 Cube 不占主导的场景**：对于 Cube 密集型算子（如大 matmul 占 >80% 时间），vector 优化收益有限
4. **连续性保证**：确认尾轴 1 的 tensor 前序是 reduce 操作（保证连续），否则 `combine_axis` 可能不生效
