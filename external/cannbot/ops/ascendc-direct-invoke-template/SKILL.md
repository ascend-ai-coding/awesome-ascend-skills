---
name: external-cannbot-ops-ascendc-direct-invoke-template
description: Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的样例工程和清晰的修改指南。触发：当用户需要创建
  Kernel 直调工程、学习 Ascend C 编程、快速原型验证、或提及"Kernel直调"、"<<<>>>内核调用"时使用本 skill。
original-name: ascendc-direct-invoke-template
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# Ascend C Kernel 直调工程

本 skill 提供三条独立样例，使用流程固定：**先按算子类型路由到对应模板，再按模板内 `[MODIFY]` 注释改造**。

## 场景路由

先用下表选定目标模板，再跳到 [使用方法](#使用方法) 对应分支：

| 算子类型 | 选择模板 | 典型特征与适用算子 |
|---------|---------|---------|
| **Vector**（A 分支） | `references/add_custom/` | 只用 UB / vector 单元；CopyIn→Compute→CopyOut 三段式。适用：add、mul、relu、softmax、layernorm 等逐元素 / 归约 / 广播算子 |
| **Matmul / Cube**（B 分支） | `references/matmul_custom/` | 仅适用dav-3510平台；依赖 Cube 单元；L1/L0A/L0B/L0C 四级缓存；需 tiling + scheduler。适用：matmul、GEMM、BMM、量化 matmul、matmul+bias |
| **mxfp8 matmul + eltwise 融合** | `references/matmul_fusion_guide.md` | Cube + Vector 融合的专用变种。该指南独立自洽，进入后不必再回本页 |


**不支持的场景**：Cube + Vector 混合（如 matmul+softmax、matmul+layernorm 等非 mxfp8 融合）不在本 skill 覆盖范围，请改用其他方案。

## 使用方法

### A. Vector 算子（Add 分支）

1. **复制样例目录**：
   ```bash
   # 若算子目录<your_op>未创建
   cp -r references/add_custom <your_op>
   # 若算子目录<your_op>已存在
   cp -r references/add_custom/* <your_op>
   cd <your_op>
   ```

2. **全局替换算子名**：`add_custom` → `<your_op>`（add_custom 是整体算子名，_kernel/_torch/_tiling 是固定后缀）

3. **阅读代码中的注释**（搜索 `[MODIFY]` 标记），修改以下内容：
   - 类名和 kernel 函数名
   - Tiling 结构体（`add_custom_tiling.h`）
   - 计算逻辑（`add_custom_kernel.asc`）
   - 输入/输出数量
   - `CMakeLists.txt` 中的目标名

4. **编译运行**：
   ```bash
   # 完整流程（含编译）
   bash run.sh

   # 仅运行测试，复用已有编译产物（代码审查阶段使用，避免重复编译）
   bash run.sh --skip-build
   ```
   > `run.sh` 在运行 kernel 前会自动删除旧的 `output/output.bin`，确保精度验证读取的是本次运行的新鲜输出。

### B. Matmul / Cube 算子（Matmul 分支）

1. **复制样例目录**：
   ```bash
   cp -r references/matmul_custom <your_project_name>
   cd <your_project_name>
   ```

2. **阅读 `matmul_custom.cpp` 中的 `[MODIFY]` 标记**，修改点按优先级：
   - **(1) dtype**：`AType / BType / CType`（bf16 / fp16 / fp8 / fp4…），同步 `sizeA/B/C` 字节数
   - **(2) LayoutA / LayoutB**：`RowMajor` / `ColumnMajor`；B 侧一般在 host 按 `transB` 实例化两份 kernel
   - **(3) Kernel 函数名与 CMake 目标名**：`matmul_custom` → `<your_op>_custom`（`run.sh` 中 `OP_NAME` 同步）
   - **(4) TilingData / Tiling 引擎**：`include/tiling/matmul_tiling_data.h` + `matmul_tiling.h`；新增 bias/scale 字段需同步
   - **(5) BlockMmad 特化**：通常无需改；需要量化/MX/Attention 变种时在 `include/policy/dispatch_policy.h` 加新 tag，在 `include/block/matmul_block_mmad.h` 加新 SFINAE 特化
   - **(6) gen_data.py**：生成 A/B 输入，golden 用 fp32 计算后 cast 到目标 dtype

3. **编译运行**：
   ```bash
   bash run.sh                      # 默认 256 256 256，transA=false transB=true
   bash run.sh 1024 1024 1024       # 指定 M K N
   bash run.sh 256 256 256 false false   # 指定 transA transB
   bash run.sh --skip-build 256 256 256  # 跳过编译
   ```

## 文件结构（Add 分支）

模板按职责分目录，PyTorch 对接开箱即用：

```
├── op_kernel/               NPU 计算层
│   ├── add_custom_tiling.h      Tiling 常量 + 结构体（纯 C/C++，kernel 和 host 共用）
│   └── add_custom_kernel.asc    纯 kernel 代码（KernelAdd 类 + add_custom_kernel 核函数入口）
├── op_host/                 Host 直调层
│   ├── add_custom.asc           Host + main 入口（#include "add_custom_kernel.asc"）
│   └── data_utils.h             数据读写工具
├── op_extension/            PyTorch 接入层
│   ├── add_custom_torch.cpp     PyTorch host 实现（Tiling 计算 + kernel launch）
│   ├── register.cpp             TORCH_LIBRARY 注册（含 Meta backend）
│   └── ops.h                    函数声明
├── scripts/                 测试脚本
│   ├── gen_data.py               生成输入数据
│   ├── golden.py                 Golden 计算函数（直调 & PyTorch 双通路共用）
│   ├── verify_result.py          直调通路精度验证
│   └── test_torch.py             PyTorch 通路测试
├── CMakeLists.txt           双 target：可执行文件 + libadd_custom_ops.so
├── run.sh                   一键运行（支持 --torch 跑 PyTorch 通路）
└── README.md
```

## PyTorch 对接

模板已内置 PyTorch 对接，编译后即可使用：

```python
import torch
import torch_npu

torch.ops.load_library("build/libadd_custom_ops.so")
y = torch.ops.npu.add_custom(x1, x2)
```

## 文件说明

### 通用入口
| 文件 | 说明 |
|------|------|
| `SKILL.md` | 本文件，路由与入口 |
| `references/kernel_launch_details.md` | **进阶（通用）**：内存层次、Double Buffer、同步机制、多 I/O |
| `references/matmul_custom_launch_details.md` | **进阶（Matmul 专用）**：四层架构、Cube 内存层次、SWAT tiling、serpentine 调度、ping-pong 流水、dtype 切换 |

### Matmul 分支
| 文件 | 说明 |
|------|------|
| `references/matmul_custom/matmul_custom.cpp` | 核心样例 launcher，含 `[MODIFY]` 标记 |
| `references/matmul_custom/include/kernel/matmul_kernel.h` | Kernel 模板层：GM Tensor 构造 + scheduler/mmad 驱动 |
| `references/matmul_custom/include/block/matmul_block_mmad.h` | BlockMmad：L1/L0/L0C 三级 ping-pong MMAD 流水 |
| `references/matmul_custom/include/block/matmul_block_scheduler.h` | BlockScheduler：serpentine 行遍历 |
| `references/matmul_custom/include/policy/dispatch_policy.h` | DispatchPolicy tag（扩展变种的入口） |
| `references/matmul_custom/include/tiling/matmul_tiling*.h` | Host SWAT tiling（baseM/baseN/baseK/kL1） |
| `references/matmul_custom/include/utils/matmul_constant.h` | C0 颗粒度、数据字节数、WINDOW_LEN 等常量 |
| `references/matmul_custom/common/` | host_utils（ACL 会话、IO）+ kernel_utils（layout trait、shape） |
| `references/matmul_custom/third_party/tensor_api/` | dav-3510 专用硬件 API（CopyGM2L1/CopyL12L0/Mad/fixpipe） |
| `references/matmul_custom/scripts/{gen_data,verify_result}.py` | A/B 生成 + golden + bf16 精度校验（rtol=atol=1e-3） |

## Matmul 关键模式（速查）

在 `matmul_custom.cpp` + `matmul_kernel.h` 中可直接学习：
- **四层模板堆叠**：Launcher → Kernel → BlockMmad → tensor_api
- **SFINAE dispatch**：通过 DispatchPolicy tag 在编译期分派计算变种（量化 / MX / 默认）
- **LayoutB 运行时选择**：host 按 `transB` 模板实例化两份 kernel；kernel 内 `std::conditional_t` 映射 `NDLayoutFormat` / `DNLayoutFormat`
- **三级 ping-pong**：L1 双缓冲 + L0 ping-pong + L0C ping-pong
- **fixpipe 自动 cast**：L0C 始终 fp32 累加，`CType` 推断 quantPre，bf16/fp16/fp32 写回全自动

## Add 代码关键模式（速查）

在 `add_custom_kernel.asc` 和 `add_custom.asc` 中可直接学习：

- **内存分配**: `TPipe` + `TQue` 管理 UB Buffer
- **数据流**: CopyIn → Compute → CopyOut 三段模式
- **同步**: `EnQue/DeQue` 确保操作顺序
- **Host 流程**: ACL 初始化 → KernelCall → 资源释放

## 参考资源

- [Ascend C 示例代码](https://gitcode.com/cann/asc-devkit/tree/master/examples)
- NPU 架构配置详见 `npu-arch` skill
- PyTorch 对接详见 `torch-ascendc-op-extension` skill
