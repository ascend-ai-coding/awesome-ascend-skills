# Golden 生成与精度对比（功能验收的参考实现）

本参考在测试设计、联调与功能精度验收节点按 implementation_route=tensor_api_custom 条件加载。补齐 SKILL 门禁「Golden 独立于被测 kernel」与 `launcher-development.md`「Launcher 不计算 CPU Golden」之间**「谁来算、怎么算、怎么对比」**的空隙。

## 0. 职责划分

| 角色 | 职责 |
|---|---|
| Golden 生成脚本（独立 Python/主机程序） | 用高精度参考实现，从逻辑输入算出参考输出 |
| Launcher | 只负责 H2D/D2H、按物理数据合同准备 buffer、launch，不判定精度 |
| Verifier | 读 raw output + golden，执行对比、出误差报告 |

三者分离，Golden 实现不得复用被测 kernel 的代码路径（含 kernel 侧 Tiling/公式），否则失去独立性。

## 1. Golden 生成方法（numpy 参考，官方 gen_data.py 形态）

```python
import numpy as np

rng = np.random.default_rng(0)                    # 固定 seed，保证可复现
a = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float16)   # 输入：目标 dtype
b = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float16)
bias = rng.uniform(-1.0, 1.0, (N,)).astype(np.float16)

golden = a.astype(np.float32) @ b.astype(np.float32) + bias.astype(np.float32)  # 参考：升 fp32 算
golden_final = golden.astype(np.float16)          # 最后转回目标 dtype（模拟输出舍入）

a.tofile("input/a.bin"); b.tofile("input/b.bin")  # 输入按目标 dtype 落盘
golden_final.tofile("output/golden.bin")          # golden 按目标 dtype 落盘
```

要点：
1. **固定 seed**：随机输入可复现，跨轮对比才有效。
2. **高精度参考**：中间计算升到 fp32（或更高），模拟「数学真值」后转回目标 dtype。
3. **输入 dtype 一致**：输入用目标 dtype 落盘（fp16/bf16/int8…），与 kernel 实际消费的 H2D bytes 一致。
4. **连续重复 launch 的 golden 复用**：同 shape 一次生成，多次 launch 都与之对比。

## 2. ABI 解码（量化 / 分形 / trans 输入先还原到逻辑域）

Golden 在**逻辑域**计算。当物理 buffer 与逻辑形状/数值不一致时，对比前必须按 `abi_crosswalk` 还原：

- **量化输入**（hifloat8/MXFP4/int8）：物理 buffer 是量化后整数 + scale，还原为 `逻辑值 = 量化值 × scale` 后再算 golden；或 golden 按量化公式反向对齐。量化 ABI 见 `precision-model.md` §5。
- **分形（NZ/ZN）**：物理 buffer 是重排过的（`(K/C0, M/16, 16, C0)`），还原到逻辑 ND 形状再算。
- **trans**：`transA/transB` 改变逻辑 shape，golden 用转置后的逻辑矩阵。
- **padding/尾块**：padding 区域**不参与 Golden**（见 `api-dataflow-layout.md` mmad 门禁），verifier 只对比 valid/tail range。

没有逻辑到物理转换时，逻辑 bytes 本身就是事实源，不额外伪造 witness（见 `launcher-development.md` §2）。

## 3. 精度对比：误差门 vs bit 级一致（两种语义不混用）

### 3.1 功能验收：误差门（rtol/atol）

```python
output = np.fromfile("output/output.bin", dtype=np.float16)  # 被测 kernel 输出
golden = np.fromfile("output/golden.bin", dtype=np.float16)  # 参考输出
if output.size != golden.size:
    raise SystemExit("size mismatch")
close = np.isclose(output, golden, rtol=1e-3, atol=1e-3, equal_nan=True)
error_idx = np.flatnonzero(~close)
error_ratio = error_idx.size / golden.size
print(f"error ratio: {error_ratio:.6f}")
assert error_ratio <= 1e-3
```

- 用**相对 + 绝对双门**（`rtol` + `atol`），`equal_nan=True` 让 NaN 相等（非有限值语义见下）。
- 阈值按算子 dtype/累加链取（fp16 常规 1e-3，见 `precision-model.md` §1 的 σ 模型与硬件极限量级）。

### 3.2 性能回归：bit 级一致（bitwise）

性能优化**不得改数值**，因此优化前后两次输出应**逐 bit 相同**（不是误差门）：

```python
assert (optimized_output.tobytes() == baseline_output.tobytes())  # bit 级一致
```

- 证明「优化不改算法顺序」；bit 级对比也是「配置参数无效」「构建无污染」的通用工具（`precision-model.md` §3.5）。

## 4. mismatch_count 语义（禁止混用）

门禁要求区分两类「对不上」：

| 语义 | 含义 | 判定 |
|---|---|---|
| bitwise 差异数 | 逐 bit 不同的元素数（性能回归用） | `tobytes()` 逐字节比较 |
| 超过误差门的元素数 | 误差超过 rtol/atol 的元素数（功能验收用） | `np.isclose` 取反计数 |

两者不可互相替代：bitwise 差异只对「应该 bit 一致」的回归有意义；误差门元素数只对「允许舍入误差」的功能验收有意义。报告里明确写清用的是哪个。

## 5. 非有限值与边界

- `equal_nan=True` 决定 NaN 算不算相等，按算子语义冻结（SKILL 门禁要求冻结「非有限值处理」）。
- clamp/saturation、Inf 的语义在黄金计算里显式处理，与 kernel 侧 `precision-model.md` §5 的量化约束一致。
- 连续重复 launch 用同一份 golden，验证「多轮结果稳定」（SKILL 门禁第 4 条）。

## 门禁

1. Golden 独立实现，不抄 kernel 公式/Tiling。
2. 输入用目标 dtype 落盘，golden 升 fp32 算后转回目标 dtype。
3. 量化/分形/trans 输入先 ABI 解码到逻辑域再算。
4. padding 不参与 golden。
5. mismatch 语义明确（bitwise vs 误差门）且不混用。

## 事实源

- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/custom_op_tensor_api/scripts/{gen_data.py, verify_result.py}`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时在 skill 根目录运行 `scripts/ensure_docs.py` 自动获取。