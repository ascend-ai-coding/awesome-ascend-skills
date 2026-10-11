# 常见问题速查（FAQ）

## 安装类问题

**Q: 如何安装 Triton-Ascend？**

A: 使用 pip 直接安装：`pip install triton-ascend`

**Q: 社区 Triton 和 Triton-Ascend 能否共存？**

A: 不可以。需要先卸载社区 Triton 再安装 Triton-Ascend：
```bash
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```
注意：安装依赖 Triton 的其他软件时会自动安装社区 Triton，会覆盖 Triton-Ascend，需要重新安装。

**Q: 能否在非 Ascend 硬件上使用 Triton-Ascend？**

A: 不可以，只能在 Ascend NPU 硬件环境使用。

**Q: 安装后 import triton 报错？**

A: 检查是否同时安装了社区 Triton。运行 `pip show triton` 和 `pip show triton-ascend` 确认只有一个存在。

**Q: 支持哪些 Python 版本？**

A: Python >= 3.9 且 <= 3.11。

**Q: 需要哪个 CANN 版本？**

A: CANN Community Edition 8.5.0。

## 编译类问题

**Q: kernel 编译报 MLIRCompilationError，如何定位？**

A: 设置 `MLIR_ENABLE_DUMP=1` 转储 IR，配合 `TRITON_ALWAYS_COMPILE=1` 强制重新编译。清理缓存：`rm -r ~/.triton/cache`。详见 [03-error-codes.md](./03-error-codes.md)。

**Q: 编译报 "ub overflow" 错误？**

A: kernel 中所有 tensor 的片上存储总和超出 UB 限制。A2/A3 系列为 96KB/192KB（开启/关闭 double buffer），910_95 系列为 128KB/256KB。减小 BLOCK_SIZE，减少中间变量，或将 int32 改为 int8。

**Q: 编译报 "unsupported data type uint8/fp64"？**

A: Ascend NPU 硬件不支持 uint8/uint16/uint32/uint64/fp64。替换为对应的 int 类型或 fp32。

**Q: 编译报 "failed to legalize operation 'tt.dot'"？**

A: 检查 dot 输入数据类型是否支持（fp16/fp32/bf16/int8）。添加 `al.compile_hint(a, "dot_pad_only_k")` 提示编译器。

**Q: 编译速度很慢？**

A: 设置 `TRITON_ASCEND_COMPILE_SPEED_OPT=1`，编译失败后跳过后续阶段。首次编译会缓存，后续运行会更快。

**Q: make_block_ptr 与循环/分支搭配编译失败？**

A: 这是已知限制。改用手动指针算术替代 `make_block_ptr`，或简化循环和分支结构。

**Q: 编译报 "block size must be power of 2"？**

A: BLOCK_SIZE 必须是 2 的幂。使用 `triton.next_power_of_2(n)` 获取最近的 2 的幂。

## 运行时类问题

**Q: 运行时报 "illegal memory access"？**

A: 检查 `tl.load`/`tl.store` 的 mask 是否正确设置，确保指针偏移不越界。

**Q: 运行时报 "deadlock detected"？**

A: 检查 `atomic_cas` 自旋锁是否正确释放。确保锁的获取和释放在同一执行路径上，避免条件分支导致锁未释放。

**Q: TRITON_ALL_BLOCKS_PARALLEL=1 导致死锁？**

A: 此选项要求 kernel 逻辑对执行顺序不敏感。如果 kernel 中有依赖执行顺序的操作（如原子操作、屏障），不能开启此选项。

**Q: grid > 65535 时报错？**

A: 设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 允许 grid > 65535，但需确保 kernel 逻辑对执行顺序不敏感。

**Q: tl.device_print 不输出？**

A: 需要设置 `TRITON_DEVICE_PRINT=1` 环境变量。每个线程的 GM 缓冲区最大 16KB，超限内容被丢弃。

**Q: 运行时报 "out of memory"？**

A: Global Memory 不足。减小 tensor 大小或 batch size。

## 精度类问题

**Q: NPU 运行结果与 PyTorch/CPU 参考结果不一致？**

A: 使用 `TRITON_INTERPRET=1` 在 CPU 上运行 kernel 作为精度基准。确保 kernel 内部使用 fp32 精度计算，最终写回时再转换。

**Q: fp16 计算精度不够？**

A: 在 kernel 内部使用 fp32 精度进行计算和归约，写回时 `.to(tl.float16)` 转换。NPU 上 `tl.exp` 是近似计算。

**Q: bf16 和 fp16 哪个精度更好？**

A: bf16 的动态范围更大（与 fp32 相同的指数位），但尾数精度较低。fp16 尾数精度更高但动态范围小。推荐 NPU 上使用 bf16。

**Q: matmul 结果 atol 需要设多大？**

A: bf16/fp16 输入的 matmul，atol=1e-2 是合理的。使用 `torch.testing.assert_close` 时：
```python
mask = golden.abs() < 1.0
torch.testing.assert_close(result[mask], golden[mask], atol=2**-6, rtol=0)
torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=2**-6)
```

**Q: Softmax 结果有微小差异？**

A: NPU 上 `tl.exp` 是近似计算，Softmax 的在线更新算法也有累积误差。差异在 1e-7 量级属正常。

## 性能类问题

**Q: Triton kernel 比 PyTorch 慢？**

A: 单独的简单操作（如向量加法）不会有明显加速。Triton 的优势在于融合多个操作减少内存访问。确保使用了 autotune 选择最优配置。

**Q: 矩阵乘法性能不佳？**

A: 确保使用了以下优化：
- `al.compile_hint(a, "dot_pad_only_k")` 提示
- 对角线分核策略（大矩阵）
- BLOCK_SIZE 选择 512B 对齐（如 BLOCK_M=128, BLOCK_N=256, BLOCK_K=256）
- bf16 输入类型
- `al.multibuffer` 多缓冲区

**Q: 如何查看 autotuning 结果？**

A: 设置 `TRITON_PRINT_AUTOTUNING=1`，自动调优完成后会输出每个内核的最佳配置及总耗时。

**Q: 如何查看 UB 占用量？**

A: 设置 `ENABLE_PRINT_UB_BITS=1` 获取当前 UB 占用量。设置 `TRITON_MEMORY_DISPLAY=1` 生成内存使用 json 文件。

**Q: 如何进行性能分析？**

A: 使用集成的 profiler 工具，设置 `MLIR_ENABLE_TIMING=1` 和 `LLVM_ENABLE_TIMING=1` 查看编译时间统计。

**Q: BLOCK_SIZE 如何选择？**

A: NPU 推荐 512B 对齐的 BLOCK_SIZE。fp32 数据推荐 128 的倍数，fp16/bf16 推荐 256 的倍数。使用 autotune 自动选择最优值。

## 开发贡献类问题

**Q: 如何本地构建 Triton-Ascend？**

A: 参考源码安装指南，从源码编译安装。

**Q: 提交 PR 需要通过哪些 CI 检查？**

A: CI 检查包括：编码安全与规范检查、开源片段检查、恶意代码检查、编译构建、开发者测试。

**Q: 如何从 GPU Triton 迁移算子到 NPU？**

A: 主要修改点：
1. `device='cuda'` → `device='npu'`
2. 添加 `import torch_npu`
3. 替换不支持的数据类型（uint → int，fp64 → fp32）
4. 添加 `al.compile_hint("dot_pad_only_k")` 到 dot 输入
5. 调整 BLOCK_SIZE 为 512B 对齐
6. 注意 atomic_add 限制
7. 注意 make_block_ptr 的 stride 限制

**Q: 自定义算子如何注册？**

A: 使用 `@al.register_custom_op` 装饰器，定义 `core`/`pipe`/`mode`/`symbol`/`bitcode` 属性。详见 [07-custom-op-example.md](../08-Examples-Patterns/07-custom-op-example.md)。

**Q: 如何调试自定义算子的 MLIR？**

A: 使用 `ASTSource` 和 `ttir_to_linalg` 手动编译查看 IR 输出。设置 `MLIR_ENABLE_DUMP=1` 转储中间 IR。

## GPU 到 NPU 迁移检查清单

从 GPU Triton 迁移到 NPU 时，需要逐项检查以下内容：

### 必须修改

| 检查项 | GPU 写法 | NPU 写法 | 说明 |
|--------|---------|---------|------|
| 设备指定 | `device='cuda'` | `device='npu'` | 需要 `import torch_npu` |
| 数据类型 | `uint8`, `fp64` | `int32`, `fp32` | NPU 不支持 uint 和 fp64 |
| 设备检查 | `x.is_cuda` | `x.is_npu` 或移除 | - |
| Stream | `torch.cuda.Stream()` | `torch.npu.Stream()` | - |
| 当前设备 | `torch.cuda.current_device()` | `torch.npu.current_device()` | - |

### 建议修改

| 检查项 | GPU 写法 | NPU 写法 | 说明 |
|--------|---------|---------|------|
| BLOCK_SIZE | 任意 2 的幂 | 512B 对齐的 2 的幂 | NPU 亲和 512B 对齐 |
| dot 输入 | 直接 load | load + `compile_hint("dot_pad_only_k")` | 减少不必要的 padding |
| 输出类型 | fp16 | bf16 | NPU 上 bf16 性能更优 |
| Grid 大小 | 基于 SM 数量 | 基于 AICore 数量 | 使用 `get_npu_properties()["num_aicore"]` |
| atomic_add | 多核累加 | 自旋锁 + 普通 add | NPU 不支持 atomic_add 多核累加 |

### 可选优化

| 检查项 | 说明 | NPU 特有 API |
|--------|------|-------------|
| 对角线分核 | 大矩阵减少 Bank 冲突 | 自定义分核逻辑 |
| MultiBuffer | 存算并行 | `al.multibuffer(tensor, 2)` |
| CV 同步 | Cube+Vector 协作 | `al.sync_block_set/wait/all` |
| 额外缓冲区 | 自定义算子 scratch buffer | `extra_buffers` 属性 |
| 编译提示 | 指导编译器优化 | `al.compile_hint(tensor, hint_name)` |

## 版本兼容性

| Triton-Ascend 版本 | Python 版本 | CANN 版本 | 硬件产品 |
|-------------------|------------|----------|---------|
| 3.2.0 | >=3.9, <=3.11 | CANN 8.5.0 | Atlas A2/A3/A3(910_95) |

## NPU 硬件特性速查

| 特性 | Atlas A2 | Atlas A3 (910_95) |
|------|---------|---------|
| AI Core 数量 | 可查询 `get_npu_properties()["num_aicore"]` | 可查询 `get_npu_properties()["num_aicore"]` |
| UB 大小 | 96KB（double buffer）/ 192KB（单 buffer） | 128KB（double buffer）/ 256KB（单 buffer） |
| L0C 大小 | 128KB | 256KB |
| 支持数据类型 | int8/16/32/64, fp16/32, bf16, bool | int8/16/32/64, fp16/32, bf16, bool, fp8 |
| 不支持数据类型 | uint8/16/32/64, fp64, fp8 | uint8/16/32/64, fp64 |
| Cube 单元 | 支持 fp16/bf16/int8 矩阵乘 | 支持 fp16/bf16/int8/fp8 矩阵乘 |
| L0C -> UB 通路 | 不支持 | 支持（通过 FixPipe） |
| multibuffer 默认 | 开启 | 关闭 |
| SIMT 模式 | 不支持 | 支持 |
| 512B 对齐 | 推荐 | 推荐 |

## 调试速查流程图

```
遇到问题
├── 安装问题
│   ├── import 报错 → 检查 Triton/Triton-Ascend 冲突
│   ├── 设备不可用 → 检查 torch_npu 和 CANN 安装
│   └── 版本不兼容 → 检查 Python/CANN/Triton-Ascend 版本
├── 编译问题
│   ├── MLIRCompilationError → MLIR_ENABLE_DUMP=1 查看 IR
│   ├── UB overflow → 减小 BLOCK_SIZE 或减少中间变量
│   ├── 不支持的数据类型 → 替换为支持的类型
│   └── make_block_ptr 失败 → 改用手动指针算术
├── 运行时问题
│   ├── 非法内存访问 → 检查 mask 和指针偏移
│   ├── 死锁 → 检查 atomic_cas 和屏障
│   ├── 超时 → 检查死循环或死锁
│   └── OOM → 减小 tensor 大小
└── 精度问题
    ├── 与参考结果不一致 → TRITON_INTERPRET=1 验证
    ├── fp16 精度不够 → kernel 内部使用 fp32
    ├── Softmax 差异 → 检查减最大值和 exp 近似
    └── matmul 差异 → 检查累加器精度和 atol/rtol 设置
```

## 相关文档

- [01-api-support-matrix.md](./01-api-support-matrix.md) - API 支持矩阵
- [02-data-type-matrix.md](./02-data-type-matrix.md) - 数据类型支持矩阵
- [03-error-codes.md](./03-error-codes.md) - 错误码参考
- [04-env-variables.md](./04-env-variables.md) - 环境变量完整参考
- 源码参考：[FAQ.md (zh)](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/FAQ.md)
