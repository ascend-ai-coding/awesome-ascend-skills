---
name: external-gitcode-ascend-vector-triton-ascend-ops-optimizer
description: 昇腾（Ascend） NPU 上 Triton 算子深度性能优化技能（Skill），致力于实现用户要求的 Triton 算子性能提升。核心技术包括但不限于
  Unified Buffer (UB) 容量规划、多 Tokens 并行处理、MTE/Vector 流水并行、mask（掩码）优化等。当用户提及以下内容时，务必触发此技能（Skill）：昇腾（Ascend）NPU
  上 Vector 类 Triton 算子性能优化。
original-name: vector-triton-ascend-ops-optimizer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-03-26'
synced-commit: 0a97c6e3999cf97425ca5ab07678e48089d79ff5
license: UNKNOWN
---


# Vector 类 Triton 算子性能优化

## 目标与概述

昇腾（Ascend）NPU 上 Vector 类 Triton 算子的深度性能优化专家。

**核心目标**：将指定的 Triton 算子性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好，追求极致性能。

**工作模式**：单算子优化模式。**禁止使用入图方式**来提升性能（模型侧会通过整网入图或 Piecewise 方式进行图优化，这里只关注单算子的独立优化）。

**工作原则**：
- **正确性优先**：每次修改后都必须进行正确性验证和性能测量
- **目标导向**：性能提升未达到目标前，持续优化，不停止迭代
- **迭代优化**：可以反复修改、测试、迭代，直至达成目标。修改 Triton 算子源代码前，务必备份，以便需要时恢复。
- **精准修改**：追求“手术级”的精准修改，避免引入新问题。

## 工作流程

0. 在昇腾 NPU 环境中，执行以下命令完成**环境配置**：`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh`

1. **基线性能验证**：首先，深入分析算子的输入参数、数据类型、Shape 范围、功能逻辑、计算流程及输出结果；然后，运行功能测试文件 `python -m pytest test_<op_name>.py`，验证算子的正确性和精度；最后，执行以下性能测试命令：`msprof op --output=<用户指定的路径> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py`，输出中的 Task Duration(us) 即为当前算子的耗时，将其记录为基线性能数据。

2. **深度性能优化**：根据基线分析结果，对 `<op_name>.py` 算子进行针对性优化，确保性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好，追求极致性能。需运行以下测试：
    - 性能测试（与基线对比）：`msprof op --output=<用户指定的路径> --kernel-name="<op_name>_kernel" --warm-up=20 --launch-count=20 python test_<op_name>_perf.py`
    - 正确性验证：`python -m pytest test_<op_name>.py`

3. **迭代调优过程中按需参考的文档**：`references/hardware_constraints.md`、`references/troubleshooting.md`

## 性能优化参考

1 - Ascend NPU 在架构上访存能力相对较弱，而计算能力较强，因此在设计时需要尽可能减少频繁的内存访问。**首要的关键优化点是批量处理多个 Tokens**，必须优先思考和调试，从而避免因逐个加载而产生的大量访存开销；由于受限于硬件内存容量，无法一次性处理完整的序列，仍需采用**分批次**计算。

一次循环里能处理的**最大 Token 数 N**，由 Kernel 内 **UB 可用容量**决定。**设：**
- 单 Kernel 内 UB 总容量为 **192 KB**
- 为留安全余量，仅使用 170 KB 的 **50%**（为确保启用 Double Buffering），即 **85 KB**
- 单个 Token 在 Kernel 内同时占用的 UB 空间峰值为 $S_{\text{token}}$（包含所有 load、中间变量的内存占用）

则需满足：$N \times S_{\text{token}} \le 85 \times 1024$；因此：$N \le \frac{85 \times 1024}{S_{\text{token}}}$
**示例：** 若 Kernel 只做一次 load 和一次 store，加载形状为 `(batch_size, hidden_size)` 的 **BF16** Tensor（每元素 2 Bytes），且不引入其他中间变量，则单个 Token 的 UB 占用峰值为：
$$
S_{\text{token}} = \text{hidden\_size} \times 2
$$

代入约束：
$$
N \times \text{hidden\_size} \times 2 \le 85 \times 1024
$$
据此计算优化后的循环次数 `reduced_loops` 以及单次循环可处理的**最大 Token 数 N**。单次循环应尽可能占满 UB，但需控制在 UB 大小的约一半以内，以利用 `Double Buffering` 机制实现流水并行。计算最大处理量时应使用**整数除法**（//）而非 `tl.cdiv`，否则易引发 UB 溢出问题。

2 - 掩码（mask）与尾块处理：每次核函数加载和存储 tensor 时都需使用 `mask` 来处理不需要计算的尾块。经过 mask 处理后，每个核上的 tensor 形状保持一致。

3 - 减少 kernel 内 Scalar 运算：将与 pid 或循环变量无关的计算移至辅助函数或循环外部；能合并的计算尽量合并，减少冗余操作。

4 - 对于 `index_select` 这类涉及非连续地址访问的操作，只能通过循环逐行读取数据；否则会引入大量标量（Scalar）计算（计算二维 mask），严重影响性能。

5 - 加载与计算交织：当需要多次从同一全局内存地址加载数据并进行计算（如加法）时，需采用 “加载一次、计算一次” 的方式，而不是全部加载完再统一计算。后者会导致计算流水线等待所有 tensor 加载完成，效率较低；前者可有效隐藏访存延迟。

6 - 若存在多个写入流，建议边计算边写入数据。写入流通常不会相互冲突，计算完提前写入可以增大并行的可能，提升整体性能。

7 - 使用 `tl.arange` 可以高效地生成二维 tensor 的索引，避免直接从全局内存（Global Memory，GM）中读取离散行数据进行二维数组运算所带来的大量 Scalar 计算，从而显著提升性能。

8 - 尽量避免使用 `tl.where`，因其主要适用于离散数据处理，性能较差。

9 - 避免对同一 tensor 多次调用 `insert_slice`，以提升执行效率。

10 - 执行规约操作时，优先选择最大的维度进行规约，有助于提升性能。

11 - **kernel 入参**：对于同一模型调用期间保持不变的参数，推荐声明为 `tl.constexpr` 编译期常量，以便编译器进行更好的优化；对于可能变化的参数（如 `batch_size`、`seq_len` 等），则应使用普通动态参数传入，避免过多编译期常量导致编译时间过长。

## 需遵循的规则和约束

### 单算子模式

单个算子只关注单算子模式下的基础功能和性能，**禁止使用入图方式提升性能**，因为模型侧会以整网入图或分段（Piecewise）方式对多算子进行图优化。

### tl.load 与 mask 使用要求

- 尽量合并相同 load、计算和 store 操作。例如，利用 `tl.load` 与 mask 参数，可一次性加载多个 Tokens 的数据，避免多次独立的 load。减少此类冗余操作有助于提升性能。
- 避免在 `tl.load` 中使用 other 参数，因为其内部会触发 `tl.where`，导致 load 后无法与其他 load 并行。
- 推荐的替代方案：先执行无掩码的 `tl.load`，再通过 `tl.where` 与 mask 组合实现掩码逻辑；当访问内存规则连续时，用 `tl.insert_slice` 代替。

### 分支与编译约束

在 kernel 内部的 `if-else` 分支中，同名变量的 Shape 必须一致，否则会导致编译错误。

### 数据搬运注意事项

- 保证 tl.load 加载的是连续的多行数据；若数据分布离散，需逐行加载。
- 传递给 Triton 算子的 tensor 必须是内存连续的，必要时可通过 `.contiguous()` 方法确保。
- 避免复用 `tl.load` 和 `tl.store` 的变量名，使用不同变量名可提高代码可读性，并减少数据流错误的风险。

## 执行要求

在 Ascend NPU 算子优化中，需自主完成从代码修改（追求 “手术级” 精准）、测试验证到性能对比的全流程闭环，确保性能提升达到用户要求的 **x 倍** 以上。通过迭代优化，在不引入错误的前提下，持续改进直至达标。

## 结果报告

性能优化目标真正达成后，需准确输出标准化报告：
```
## 优化结果报告

### 算子信息

- 算子名称：<op_name>
- 源文件：<file_path>

### 性能对比

| 基线耗时 (us) | 优化后耗时 (us) | 加速比 |
|-------------|---------------|-------|
| ... | ... | ...x |

### 优化技术清单

1. [已应用] 多个 Token 并行处理：N = ...
2. [已应用] 消除带 other 的 tl.load
3. ...

### 关键修改说明

- 修改点 1：...
- 修改点 2：...
```
