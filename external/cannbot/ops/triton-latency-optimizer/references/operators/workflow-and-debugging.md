# Triton-Ascend 验证与调试工作流

> 跨算子通用的验证、指标对比和调试经验，从三个源文件中提取并整合。

---

## 1. 多 case 验证必须复制 .json 文件

当任务描述使用 `get_input_groups()` 时，`.py` 会通过 `os.path.dirname(__file__)` 读取同目录下的 `{op_name}.json`。在 Phase 4 验证目录创建 `{op_name}_torch.py` 时，必须同时复制 `.json` 文件，否则 verify 会失败。

```bash
cp {workdir}/{op_name}.json {verify_dir}/{op_name}_torch.json
```

---

## 2. benchmark.py L1 闸门命名规则

`benchmark.py` 会根据 `--triton_impl_name` 查找对应的 verify_result 文件：

| triton_impl_name | 对应 verify json |
|-----------------|-----------------|
| `triton_ascend_impl` | `verify_result.json` |
| `triton_baseline` | `verify_result_baseline.json` |
| `triton_optimized` | `verify_result_optimized.json` |

由于 `verify.py` 总是输出 `verify_result.json`，在 Phase 4 优化验证后需要手动复制：

```bash
cp verify_result.json verify_result_optimized.json
```

---

## 3. 精度阈值与数据类型

比较前会统一升到 float32，避免低精度 dtype 自身误差污染。

| dtype | MERE 阈值 | MARE 阈值 |
|-------|-----------|-----------|
| float16 | 2^-10 ≈ 9.77e-4 | 9.77e-3 |
| bfloat16 | 2^-7 ≈ 7.81e-3 | 7.81e-2 |
| float32 | 2^-13 ≈ 1.22e-4 | 1.22e-3 |

---

## 4. 验证必须读取 `verify_result.json`

无论 console 输出看起来如何，判定通过与否必须读取 `verify_result.json` 中的数值字段：

```python
passed_cases == total_cases and total_cases > 0
```

多 shape 场景下，"大部分通过"不等于通过。失败 case 的清单在 `failures` 字段中。

---

## 5. 评估指标选择

- `speedup_vs_torch` 容易受 framework 延迟抖动影响，尤其当框架延迟本身很小（<0.1ms）时，几次测试的波动就能让比值变化几个点。
- 对于内存瓶颈算子，以及 framework 参考实现已高度优化的算子（如 `torch_npu.npu_swiglu_quant`），**implementation 平均延迟**是更稳定的优化指标。
- 多 shape 场景以几何平均加速比或实现延时几何平均为准，避免被个别异常 shape 带偏。

---

## 6. 何时停止优化

当出现以下情况时，通常可以进入终局：

- `latency-optimizer` 技能清单中已无命中条件成立的优化点。
- 每次新优化带来的实现延迟提升 < 1%，且伴随编译/验证风险上升。
- 进一步优化需要改变数据布局或引入复杂临时缓存，会触及正确性边界。

**原则**：在精度正确的前提下追求性能；当边际收益趋近于零时，保留已验证的最佳版本。

---

## 7. 验证与基线管理

- 每次优化后必须跑完整 `verify.py`，读取 `verify_result.json` 的 `passed_cases == total_cases`。
- Phase 4 与 Phase 3 基线对比时，使用几何平均 `speedup_vs_torch`；基线 verify 结果直接复用，不必重跑。
- 同一时段重新跑基线，排除 framework 测量抖动。
- 失败时先分类（A/B/C）：
  - **A 类**：代码逻辑/算法错误，可修复后重试。
  - **B 类**：环境/基础设施错误，及时止损。
  - **C 类**：同一子类型连续 ≥3 次重复失败，及时止损。

---

## 8. AST 预检查

每次修改 Triton 代码后，先用 `validate_triton_impl.py` 检查是否有 PyTorch fallback：

```bash
python3 validate_triton_impl.py <generated_code.py>
```

若出现 PyTorch 退化（Type1/2/3），必须在进入功能验证前修复。

---

## 9. 性能对比最佳实践

1. 优化前与优化后在同一时段测试，避免 framework 抖动。
2. 若 `speedup_vs_torch` 因 framework 抖动而下降，但 implementation latency 稳定下降，则按实现延时判定为有效优化。
3. 多 shape 全量执行：每个 shape 独立失败都不能忽略；一个 UB overflow 即整体失败。
4. 记录每轮迭代的 key numbers，便于后续复盘。
