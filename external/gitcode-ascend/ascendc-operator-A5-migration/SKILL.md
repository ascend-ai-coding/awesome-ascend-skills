---
name: external-gitcode-ascend-ascendc-operator-A5-migration
description: AscendC 算子从 A2/A3(910b/910_93) 迁移到 A5(950) 的 L1+L2+L3 级别改造。当用户需要对 AscendC
  算子做 950 适配、A5 迁移、ascend950 编译、算子跨芯片移植、MicroAPI 重写、RegBase 改造、SIMT 优化时使用。触发词：迁移、950、A5、RegBase、arch35、L1、L2、L3、MicroAPI、RegTensor、CastTrait、SIMT、Scatter、Gather。
original-name: ascendc-operator-A5-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-06-14'
synced-commit: 62f55373f87e10e1d64f3f82f369be625c56b9fc
license: UNKNOWN
---

# AscendC A2/A3 → A5 算子迁移 (L1+L2+L3)

## 迁移层级决策树

```
算子是否满足以下任一条件？
  ├─ 性能关键路径 (RMSNorm/RoPE 等高频算子)
  ├─ 量化 Cast 链路复杂 (FP32→FP8/HiFloat8/INT8)
  ├─ 需要溢出模式控制 (RMSNorm/Softmax)
  └─ 950 新增数据类型 (FP8/HiFloat8 的 Cast 需 MicroAPI)
  │
  ├─ 满足任一 → L2：RegBase API 重写（MicroAPI）
  │
  └─ 不满足，是否满足以下全部？
      ├─ 包含 Scatter/Gather 操作（按索引读写 GM）
      ├─ 索引逻辑简单（无需复杂计算）
      ├─ 无需 UB 中转（SIMT 可直接访问 GM）
      └─ 线程并行度高（数据量足够大）
      │
      ├─ 全部满足 → L3：SIMT 优化（在 L1 基础上新增 SIMT kernel）
      │
      └─ 不满足 → L1：基础适配（所有算子必做）
```

### 层级判定核心原则

| 层级 | 本质 | 改动范围 |
|------|------|----------|
| **L1** | 950 基础适配：独立配置 + RegBase kernel 入口 + BF16 条件编译移除（所有算子必做） | `_def.cpp` + `_apt.cpp` + `arch35/` + `CMakeLists.txt` + `config/` |
| **L2** | 核心计算路径全部重写为 MicroAPI Register-based 模式（在 L1 基础上） | 全部重写 |
| **L3** | Scatter/Gather 操作用 SIMT 多线程替代，零 UB 占用（在 L1 基础上） | `arch35/` 新增 SIMT kernel |

### L4 信号（超出本 Skill，提示用户升级）

- Tiling 需要 `IsRegbaseSocVersion` 判断 → L4
- UB 预留不足（需 40KB SIMT DCache） → L4

## 工作流概览

0. **阶段 0：环境信息获取与验证** → 产出：完整环境信息记录（CANN + Python + NPU）
1. **阶段 1：评估定级** → 产出：算子迁移层级判定 + 算子源码分析摘要
2. **阶段 2：代码改造** → 产出：修改后的代码文件
3. **阶段 3：编译验证** → 产出：编译结果 + 错误修复
4. **阶段 4：精度验证** → 产出：精度测试用例 + 精度测试报告 + 精度闭环确认
5. **阶段 5：结果评估** → 产出：迁移结论 + 升级建议

## 阶段 0：环境信息获取与验证（MANDATORY — 所有阶段的前置）

**本阶段是整个迁移流程的第一步，必须在任何其他操作之前完成。无论用户从哪个阶段开始（首次执行、迁移后执行、重新打开会话测试精度等），都必须先检查环境信息完整性。**

### 0.1 全流程环境验证机制

**在 Skill 执行的任意步骤启动时，MUST 先执行以下环境检查**：

```
任意步骤启动
  │
  ├─ 检查：环境信息是否已获取且完整？
  │   ├─ 否（首次执行 / 信息缺失）→ 执行 Step 0.2 获取完整环境信息
  │   └─ 是 → 检查：环境信息是否可能已变更？
  │       ├─ 是（如检测到 Python 路径变化、CANN 版本变化）→ 提示用户确认或重新提供
  │       └─ 否 → 直接复用已保存的环境信息
  │
  └─ 环境信息确认完整 → 继续执行当前步骤
```

**环境信息缺失时的处理**：
- **MUST 立即暂停当前操作**，不得跳过环境检查直接执行
- 以清晰、友好的方式向用户请求确认相关信息
- 仅在获取到完整且有效的环境信息后，方可继续执行后续操作

### 0.2 环境信息获取

**MUST 向用户获取以下环境信息。对于每项信息，优先自动检测，检测不到时再向用户询问。**

#### 0.2.1 CANN 环境信息

| 信息项 | 自动检测方式 | 询问话术（自动检测失败时） |
|--------|------------|--------------------------|
| CANN 安装路径 | `echo $ASCEND_HOME_PATH` | "请提供 CANN 安装路径（如 `/usr/local/Ascend/ascend-toolkit`）" |
| CANN 版本号 | `cat $ASCEND_HOME_PATH/ascend_toolkit_install.info 2>/dev/null \|\| echo unknown` | "请提供 CANN 版本号（如 9.0.0）" |
| `set_env.sh` 路径 | `$ASCEND_HOME_PATH/../set_env.sh` 或 `$ASCEND_HOME_PATH/set_env.sh` | "请提供 CANN 环境激活命令路径（如 `source /path/to/set_env.sh`）" |

#### 0.2.2 Python 环境信息

| 信息项 | 自动检测方式 | 询问话术（自动检测失败时） |
|--------|------------|--------------------------|
| Conda 环境名 | `echo $CONDA_DEFAULT_ENV`（非 `base` 且非空时有效） | "请提供 Conda 环境名称（如 `my_env`）" |
| Python 解释器路径 | `which python3` | "请提供 Python 解释器绝对路径（如 `/home/user/miniconda3/envs/my_env/bin/python`）" |
| Python 版本 | `python3 --version` | "请提供 Python 版本（如 3.10）" |
| torch 版本 | `python3 -c "import torch; print(torch.__version__)"` | "请确认 torch 是否已安装及其版本" |
| torch_npu 版本 | `python3 -c "import torch_npu; print(torch_npu.__version__)"` | "请确认 torch_npu 是否已安装及其版本" |
| pytest 版本 | `python3 -c "import pytest; print(pytest.__version__)"` | "请确认 pytest 是否已安装及其版本" |

#### 0.2.3 NPU 硬件信息

| 信息项 | 自动检测方式 | 询问话术（自动检测失败时） |
|--------|------------|--------------------------|
| NPU 可用性 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` | "请确认 NPU 设备是否可用" |
| NPU 设备数量 | `python3 -c "import torch_npu; print(torch.npu.device_count())"` | — |
| 可用设备编号 | `npu-smi info 2>/dev/null \|\| echo "npu-smi not found"` | "请提供可用的 NPU 设备编号（如 0、1、2 等）" |

### 0.3 环境信息记录格式

获取完成后，**MUST 将以下信息记录到当前会话上下文中**，供所有后续步骤引用：

```
## 环境信息记录（会话持久保存）

### CANN 环境
- CANN_PATH: <cann_install_path>
- CANN_VERSION: <version>
- SET_ENV_CMD: source <cann_path>/set_env.sh

### Python 环境
- CONDA_ENV: <env_name> (如无 conda 则记录 "N/A")
- PYTHON_PATH: <python_abs_path>
- PYTHON_VERSION: <version>
- TORCH_VERSION: <version>
- TORCH_NPU_VERSION: <version>
- PYTEST_VERSION: <version>

### NPU 硬件
- NPU_AVAILABLE: True/False
- NPU_DEVICE_COUNT: <count>
- NPU_VISIBLE_DEVICE: <device_id>  (用户指定或自动选择)

### 关键路径（从环境信息推导）
- CANN_INCLUDE: <CANN_PATH>/aarch64-linux/include
- CANN_ACLNN_INCLUDE: <CANN_PATH>/aarch64-linux/include/aclnn
- TORCH_NPU_PATH: <python -c "import os,torch_npu;print(os.path.dirname(torch_npu.__file__))">
- TORCH_CMAKE_PREFIX: <python -c "import torch;print(torch.utils.cmake_prefix_path)">
```

### 0.4 环境信息复用规则

| 场景 | 处理方式 |
|------|---------|
| 同一会话内再次需要环境信息 | 直接复用已保存的信息，无需重复获取 |
| 检测到环境信息可能变更（如 Python 路径变化） | 提示用户确认："检测到 Python 路径可能已变更（原: xxx, 当前: yyy），是否使用新路径？" |
| 用户主动提供新环境信息 | 更新已保存的环境信息记录 |
| 新会话（环境信息为空） | 执行完整的环境信息获取流程 |

### 0.5 环境验证脚本

**获取环境信息后，MUST 执行以下验证脚本确认环境可用**：

```bash
source ${SET_ENV_CMD#source } && ${PYTHON_PATH} -c "
import torch
import torch_npu
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
if torch.npu.is_available():
    x = torch.randn(4, 4, device='npu:0')
    print(f'NPU tensor OK: {x.device}')
    print(f'NPU device count: {torch.npu.device_count()}')
else:
    print('WARNING: NPU not available, check npu-smi info')
"
```

### 0.6 常见环境问题

| 问题 | 报错信息 | 解决方案 |
|------|---------|---------|
| 未 source CANN | `Cast ADD_TO_LAUNCHER_LIST_AICORE failed` | 执行 `${SET_ENV_CMD}` |
| torch_npu 未安装 | `ModuleNotFoundError: No module named 'torch_npu'` | 安装对应 CANN 版本的 torch_npu |
| NPU 设备未就绪 | `NPU available: False` | 检查 `npu-smi info` 是否正常 |
| CANN 版本与 torch_npu 不匹配 | `RuntimeError: aclnn execute failed` | 确保 CANN 和 torch_npu 版本对应 |
| NPU 设备被占用 | 创建 tensor 时卡住 | 设置 `ASCEND_RT_VISIBLE_DEVICES` 指定空闲设备 |

### 检查点

- [ ] CANN 安装路径和版本已确认
- [ ] Python 解释器路径和版本已确认
- [ ] torch / torch_npu / pytest 版本已确认
- [ ] NPU 可用性和设备编号已确认
- [ ] 环境验证脚本执行成功
- [ ] 环境信息已记录到会话上下文

**全部通过 → 进入阶段 1**

## 阶段 1：评估定级

### 必须检查的文件（按优先级）

| 文件 | 检查内容 | 判定影响 |
|------|----------|----------|
| `_def.cpp` | SoC 列表、数据类型 | 是否需要扩展数据类型 → L2 |
| kernel `.cpp` | `__NPU_ARCH__` 条件编译、BF16 保护 | L1 必须移除 BF16 保护 |
| kernel `.h` | 是否使用 `LocalTensor`/`DataCopy`/`Cast`+`RoundMode` | 是否需要 MicroAPI 重写 → L2 |
| `op_host/CMakeLists.txt` | 是否已有 ascend950 分支 | 编译选项是否就绪 |
| Tiling `.cpp/.h` | `IsRegbaseSocVersion`、UB 预留 | 是否需要 Tiling 适配 → L4 |

### L2 特有判定信号

| 信号 | 说明 |
|------|------|
| kernel 中有 `Cast<fp8_e4m3fn_t, float>` 或 `Cast<hifloat8_t, float>` | 950 新增类型必须用 MicroAPI CastTrait |
| kernel 中有 `ReduceSumCustom` | 950 应替换为 `ReduceSum`（MicroAPI） |
| kernel 中有 `DataCopyPad` 非对齐存储 | 950 应替换为 `DataCopyUnAlign` |
| 算子是 RMSNorm/RoPE/Softmax | 性能关键路径，溢出模式控制可提升性能 |
| kernel 中有 FP32→INT8 量化 | 950 需三步量化 + Pack |

### L3 特有判定信号

| 信号 | 说明 |
|------|------|
| kernel 中有 Scatter/Gather 操作 | 按索引读写 GM，SIMT 多线程可替代 |
| 索引逻辑简单 | 无需复杂计算，SIMT 线程可独立处理 |
| 无需 UB 中转 | SIMT 可直接访问 GM |
| 线程并行度高 | 数据量足够大，SIMT 多线程有收益 |

### 检查点
- [ ] 确认算子数据类型（是否涉及 FP8/HiFloat8 等新类型）
- [ ] 确认 kernel 中 BF16 条件编译保护是否影响 950
- [ ] 确认 kernel 是否使用 Memory-based API（`LocalTensor`/`DataCopy`/`Cast`+`RoundMode`）
- [ ] 确认 Tiling 是否需要 UB 预留
- [ ] 确认是否有 Scatter/Gather 操作可做 SIMT 优化
- [ ] 明确标注迁移层级及理由
- [ ] **算子源码分析摘要已生成**（见下方「算子源码分析摘要」章节）

### 边缘情况

- **算子有 BF16 但无 `__CCE_AICORE__` 保护**：950 原生支持 BF16，L1 的 arch35/ 中直接使用即可
- **`op_host/CMakeLists.txt` 已有 ascend950 分支**：说明构建系统已适配，直接复用
- **算子部分路径需 MicroAPI、部分不需要**：可混合使用，MicroAPI 路径用 `__VEC_SCOPE__` 包裹

### 算子源码分析摘要（MANDATORY）

**目的**：无论算子是否有文档，都必须基于被迁移源码生成分析摘要，作为后续精度测试用例生成的依据。摘要内容以被迁移源码为准。

#### 生成流程

```
1. 读取算子源码文件（_def.cpp / kernel .cpp/.h / tiling .cpp/.h）
2. 如有算子文档/设计文档，交叉对比，记录差异点
3. 按模板输出摘要
```

#### 摘要模板

```markdown
# 算子源码分析摘要：<算子名>

## 1. 算子功能描述
- 数学公式 / 计算逻辑（从 kernel 代码逆向提取）
- 算子类别：elementwise / reduction / attention / matmul / 其他

## 2. 接口签名
- 输入参数（名称、类型、shape 约束、数据类型）
- 输出参数（名称、类型、shape 计算规则）
- 可选参数 / 属性参数

## 3. 支持的数据类型
- 从 _def.cpp 的 DataType 配置提取
- 从 kernel 中的模板特化 / 条件编译路径提取
- 列出：FP16 / BF16 / FP32 / INT8 / FP8 / HiFloat8 等

## 4. 计算逻辑伪代码
- 从 kernel .cpp/.h 的 Compute 函数提取 AscendC API 调用序列
- 标注升精度路径（FP16→FP32→FP16 等）
- 标注条件编译分支（BF16 路径 / FP32 路径等）

## 5. Tiling 策略
- Block 级切分方式（formerNum / formerLength / tailNum / tailLength）
- UB 级切分方式（tileLength 计算、bufferCoefficient）
- 特殊约束（对齐要求、尾 tile 处理）

## 6. 输入域约束
- 数学定义域（如 acosh 要求 x≥1，log 要求 x>0）
- 数据范围约束（FP16 最大 65504 等）
- Shape 约束（维度限制、对齐要求）

## 7. 边界条件与特殊值
- 零值 / 极小值 / 极大值的行为
- NaN / Inf 的传播规则
- Subnormal（非规格化数）处理方式

## 8. 文档与源码差异（如有文档时必填）
- 文档描述与源码实现不一致之处
- 文档缺失但源码实现了的功能
- 文档描述但源码未实现的功能
- **注意**：测试用例生成以源码为准，不以文档为准
```

#### 逆向分析方法（无文档算子）

| 方法 | 适用场景 | 具体操作 |
|------|---------|---------|
| **代码逆向分析** | 所有算子 | 从 kernel Compute 函数逐行提取 API 调用序列，反推数学公式 |
| **同类算子类比** | 算子名可推断功能 | 如 `fast_gelu` 类比 `gelu`，`rms_norm` 类比 `layer_norm`，补充边界值和输入域 |
| **核心功能推断** | 算子名模糊 | 从输入输出 shape 关系 + API 调用序列推断算子类别和功能 |
| **测试用例反推** | 有已有测试代码 | 从已有 UT/ST 测试用例反推算子的输入域和边界条件 |
| **PyTorch 对标** | 有 PyTorch 同名接口 | 用 `torch.<op_name>` 作为 CPU 参考实现，确认语义对齐 |

#### 有文档算子的处理

即使算子有完整文档，仍**必须**基于源码生成摘要：
1. 先按无文档流程从源码提取信息
2. 再与文档交叉对比，将差异记录到「文档与源码差异」章节
3. 测试用例生成**始终以源码为准**，文档仅作参考

## 阶段 2：代码改造

### 路由表：根据层级加载对应指南

| 层级 | 必须加载 | 补充参考文档目录 | 不要加载 |
|------|----------|-----------------|----------|
| L1 | `l1-implementation-guide.md` | `references/migration/` | `l2-register-based-guide.md` |
| L3 | `l1-implementation-guide.md` + `l3-simt-optimization-guide.md` | `references/simt/`、`references/migration/` | `l2-register-based-guide.md` |
| L2 | `l1-implementation-guide.md` + `l2-register-based-guide.md` | `references/reg-base-vector/`、`references/memory-base-vector/`、`references/datacopy/`、`references/migration/` | `l3-simt-optimization-guide.md` |

**MANDATORY - READ ENTIRE FILE**：在继续之前，你必须完整阅读对应的指南文件。

- L1：[`l1-implementation-guide.md`](references/l1-implementation-guide.md)
- L3：[`l3-simt-optimization-guide.md`](references/l3-simt-optimization-guide.md)
- L2：[`l2-register-based-guide.md`](references/l2-register-based-guide.md)

### 补充参考文档目录结构

```
references/
├── migration/                  迁移相关官方文档（兼容性说明、架构变更、API迁移指导、编译迁移指导）
├── memory-base-vector/         Memory-based Vector 操作（连续计算API、高维切分API、归约API、掩码API、同步控制、数据结构）
├── reg-base-vector/            Register-based Vector / MicroAPI（Reg矢量计算编程、C API reg_load/reg_store/reg_vector）
├── datacopy/                   数据搬运操作（DataCopy/DataCopyPad/Copy、C API vector_datamove/cube_datamove）
├── cube/                       Cube 类操作（MMAD/稀疏矩阵、C API cube_compute）
├── simt/                       SIMT 相关内容（C API、BuiltIn关键字、编程模型、线程架构、同步机制、混合编程、原子操作、缓存控制、系统变量）
├── simd/                       SIMD 相关内容（C API、BuiltIn关键字、vector_compute/vector_datamove）
├── api-overview/               API 选型与概述（编程接口概述、基础API概述、高阶API概述、Tensor API 结构体）
└── precision-testing/          精度测试相关（精度标准、aclnn接口指南、源码逆向分析、测试模板、PyTorch绑定构建）
    ├── OPS_PRECISION_STANDARDS.md              生态算子开源精度标准（MERE/MARE 阈值表）
    ├── aclnn-interface-guide.md                aclnn 接口调用指南（唯一标准方式：EXEC_NPU_CMD + torch.ops.npu）
    ├── pytorch-binding-build-guide.md          自定义算子 PyTorch 绑定构建指南（EXEC_NPU_CMD 标准方式，唯一标准）
    ├── torch_aclnn_helper.h                    CANN 标准桥接头文件（EXEC_NPU_CMD 宏 + ConvertType 体系，必须复制到绑定项目）
    ├── precision-test-pre-validation-guide.md  精度测试前置数值验证指南（小规模 NPU/CPU 对比，确保语义一致）
    ├── source-code-reverse-analysis.md         算子源码逆向分析方法论（无文档算子的用例生成策略）
    ├── test_op_precision_aclnn_template.py     aclnn 精度测试 pytest 模板
    ├── run_precision_report_aclnn_template.py  aclnn 精度报告生成器模板
    └── precision_report_template.md            精度报告 Markdown 模板
```

**绝对不要设置任何行数限制。**

**绝对不要加载** `stage_1_planning.md`、`stage_2_implementation.md`、`stage_3_testing.md`、`stage_4_evaluation.md`（模板残留，无有效内容）。

### L1 改动清单（精确步骤，所有算子必做）

#### 步骤 1：_def.cpp 独立配置

950 使用独立 `OpAICoreConfig`，通过 `opFile` 切换到 RegBase kernel：

```cpp
OpAICoreConfig regbaseCfg;
regbaseCfg.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "算子名_apt");
this->AICore().AddConfig("ascend950", regbaseCfg);
```

如果 950 需要扩展数据类型（如新增 FP8/HiFloat8），在独立配置中添加：

```cpp
regbaseCfg.Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
               ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
    .Format({...});
```

**配置层共性特征**：

| 共性特征 | 说明 |
|---------|------|
| 独立 OpAICoreConfig | 950 使用独立的 `OpAICoreConfig` |
| opFile.value 设置 | 设置 `opFile.value = "算子名_apt"` |
| DynamicCompileStaticFlag | 950 配置普遍开启动态编译静态化 |
| DynamicRankSupportFlag | 950 配置普遍开启动态 Rank 支持 |
| DynamicShapeSupportFlag | 950 配置普遍开启动态 Shape 支持 |

#### 步骤 2：创建 _apt.cpp

```cpp
// 算子名_apt.cpp — RegBase kernel 入口
#include "arch35/算子名_impl.hpp"
#include "arch35/算子名_bf16.hpp"
#include "arch35/算子名_single.hpp"

extern "C" __global__ __aicore__ void 算子名(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    // 与 910b 版本相同的逻辑
}
```

**关键差异**：
- include 路径从根目录改为 `arch35/`
- Tiling 传参可能从 `tiling` 改为 `tempTilingGm`（解析后的结构体）

#### 步骤 3：创建 arch35/ 目录

将根目录下的实现头文件复制到 `arch35/`，做以下微调：

| 调整项 | 910b 版本 | 950 arch35/ 版本 |
|--------|-----------|-----------------|
| BF16 条件编译 | `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` | 直接支持，移除条件编译 |
| BF16 架构保护 | `#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 \|\| __NPU_ARCH__ == 3113))` | 移除保护 |
| Tiling 传参 | `op.Init(..., tiling)` — 传原始 tiling 指针 | `op.Init(..., tempTilingGm)` — 传解析后的结构体 |

**BF16 条件编译移除示例**：

```cpp
// 910b 版本：
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
  BF16 路径
#endif

// 950 arch35/ 版本：
BF16 路径（直接支持，无需条件编译）
```

**BF16 架构保护移除示例**：

```cpp
// 910b 版本：
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
#endif

// 950 arch35/ 版本：
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
```

#### 步骤 4：修改 CMakeLists.txt

**决策树：是否需要修改 CMakeLists.txt？**

```
950 是否需要独立 tiling 逻辑？
  ├─ 是（IsRegbaseSocVersion 判断 / UB 预留不同 / 新增 tiling 字段）
  │   → 方案 A：显式 COMPUTE_UNIT + TILING_DIR
  └─ 否（950 与 910b/910_93 完全共用 tiling）
      → 方案 B：不修改 CMakeLists.txt，仅添加 config/ascend950/
```

**方案 A：显式 COMPUTE_UNIT + TILING_DIR**（950 需要独立 tiling 时使用）

```cmake
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE 算子名 ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
```

**方案 B：隐式自动检测**（950 与 910b/910_93 共用 tiling 时使用）

```cmake
# CMakeLists.txt 无需任何修改，保持原始内容
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE 算子名 ACLNNTYPE aclnn_exclude DEPENDENCIES ...)
# 构建系统通过 config/ascend950/<算子名>_binary.json 自动检测 950 编译支持
# kernel_src_copy 自动复制 op_kernel/arch35/ 目录到构建目录
```

**注意**：如果 `op_host/CMakeLists.txt` 中已有 ascend950 的 `add_ops_compile_options` 分支，直接复用，不要重复添加。

### CMakeLists.txt 经验教训（踩坑汇总）

**1. 禁止对同一算子调用两次 `add_modules_sources`**

`add_modules_sources` 会将算子名追加到全局 `COMPILED_OPS` / `COMPILED_OP_DIRS`（CACHE FORCE），两次调用导致路径重复，后续 `generate_bin_scripts` 生成同名 target 报冲突。

```cmake
# ❌ 错误：两次调用
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude)           # 原始
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude COMPUTE_UNIT "ascend950" TILING_DIR "arch35")  # 新增

# ✅ 正确：合并为一次调用
add_modules_sources(... OPTYPE foo ACLNNTYPE aclnn_exclude COMPUTE_UNIT "ascend950" TILING_DIR "arch35")
```

**2. `COMPUTE_UNIT` 与 `TILING_DIR` 列表长度必须严格相等**

`find_value_by_key()` 内部校验两个列表长度，不等则 `FATAL_ERROR`。CMake 列表中空字符串 `""` 会被自动吞掉，导致长度不匹配：

```cmake
# ❌ 错误：空字符串被吞，TILING_DIR 实际只有 1 个元素 ["arch35"]
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   ""           ""              "arch35"

# ✅ 正确：950 需要独立 tiling 时，910b/910_93 也必须放入命名子目录
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   "default"    "default"      "arch35"
# 对应 op_host/default/ 和 op_host/arch35/ 两个 tiling 目录
```

**3. `config/ascend950/` 是 950 编译的必要条件**

构建系统通过 `op_host/config/<compute_unit>/<op_name>_binary.json` 判断算子是否支持该 SoC。缺失则 950 编译被**静默跳过**（无报错），极易遗漏。

**4. `kernel_src_copy` 自动复制 `arch35/` 目录**

`kernel_src_copy()` 使用 `find -mindepth 1 -maxdepth 1 -exec cp -r` 递归复制 `op_kernel/` 下所有一级子项，`arch35/` 目录会被自动复制到构建目录，无需在 CMakeLists.txt 中显式声明。

**5. `TILING_DIR` 的语义**

`TILING_DIR` 指定 `op_host/` 下的子目录名，`add_tiling_sources()` 会同时在 `op_host/` 根目录和 `op_host/${tiling_dir}/` 查找 `*_tiling*.cpp`。当 `TILING_DIR` 为空时，只在根目录查找。

#### 步骤 5：创建 config/ascend950/

从 `config/ascend910b/` 复制两个文件：

```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<算子名>_binary.json op_host/config/ascend950/
cp op_host/config/ascend910b/<算子名>_simplified_key.ini op_host/config/ascend950/
```

**binary.json**：定义算子的二进制 kernel 编译信息（输入输出数据类型、格式等）。
**simplified_key.ini**：定义 opc 工具编译时的 `--simplified_key_mode` 选项值。

### L3 改动清单（精确步骤）

L3 在 L1 基础上（`_def.cpp`/`_apt.cpp`/`arch35/`/`config/`），**新增 SIMT kernel 替代 Scatter/Gather**：

1. 在 `arch35/` 中创建 SIMT 核函数文件（如 `xxx_simt_op.h`）
2. SIMT 核函数标记 `__simt_vf__` + `LAUNCH_BOUND(2048)`
3. 在 SIMT 函数内直接访问 `__gm__` 指针，用 `Simt::GetThreadIdx/GetThreadNum` 分配数据
4. 在 `_apt.cpp` 中用 `__NPU_ARCH__ == 3510` 条件编译切换 SIMT/Memory-based
5. 通过 `Simt::VF_CALL<SimtFunc>` 启动 SIMT 计算

### L2 改动清单（精确步骤）

L2 在 L1 基础上（`_def.cpp`/`_apt.cpp`/`arch35/`/`config/`），**核心计算路径全部重写**：

1. 引入 MicroAPI 命名空间（`RegTensor`/`MaskReg`/`CastTrait` 等）
2. 将 `LocalTensor` → `RegTensor`，`DataCopy` → `DataCopy<LoadDist/StoreDist>`
3. 将 `Cast<T1, T2>(dst, src, RoundMode, count)` → `Cast<T1, T2, CAST_TRAIT>(dst, src, maskReg)`
4. 将向量计算 API 加 `MicroAPI::` 前缀，count 参数 → maskReg
5. 将 `SetFlag/WaitFlag` → `LocalMemBar<MemType::UB>`
6. 用 `__VEC_SCOPE__` 包裹核心计算
7. 如需溢出模式控制，添加 `GetCtrlSpr/SetCtrlSpr`（必须保存/恢复）
8. 如需量化，实现三步量化 + Pack + `DataCopyUnAlign`

### 检查点
- [ ] `_def.cpp` 中 ascend950 独立配置正确（`OpAICoreConfig` + `SetOpFile`）
- [ ] `_apt.cpp` 创建完成，include 路径指向 `arch35/`
- [ ] `arch35/` 目录创建完成，BF16 条件编译已移除
- [ ] `CMakeLists.txt` 已增加 `SUPPORT_COMPUTE_UNIT "ascend950"` + `SUPPORT_TILING_DIR "arch35"`
- [ ] config 目录文件齐全（`binary.json` + `simplified_key.ini`）
- [ ] L3 场景下 SIMT 核函数已标记 `__simt_vf__` + `LAUNCH_BOUND`
- [ ] L3 场景下 `_apt.cpp` 中 `__NPU_ARCH__ == 3510` 条件编译切换正确
- [ ] L2 场景下核心计算已用 `__VEC_SCOPE__` + MicroAPI 重写
- [ ] L2 场景下 CastTrait 的 SatMode 正确（量化 SAT，反量化 NO_SAT）
- [ ] L2 场景下溢出模式控制已保存/恢复

## 阶段 3：编译验证

### 精确步骤

1. **环境确认**：MUST 先检查阶段 0 的环境信息是否完整（参见阶段 0）
2. 设置 CANN 环境：`source ${SET_ENV_CMD#source }`
3. 执行编译（**注意 `--soc` 是双横线**）：
   ```bash
   bash build.sh --pkg --ops=<算子名> --soc=ascend950 2>&1 | tee build.log
   ```
3. 如编译失败，在日志中搜索错误：`grep -n "[Ee]rror" build.log`
4. 根据错误速查表定位修复

### 常见编译错误速查

| 错误特征 | 根因 | 修复方法 |
|----------|------|----------|
| `Exec format error: bisheng` | `build/gen_bisheng_dir/bisheng` 脚本缺少 `#!/bin/bash` | 在首行加 `#!/bin/bash`（**每次 clean build 后需重新修复**） |
| `OSError: [Errno 8]` | 同上 bisheng shebang 问题 | 同上 |
| `Error 137` | 编译进程被 OOM Kill | 减少并行编译线程 `-j4` |
| BF16 编译错误 | arch35/ 下 BF16 条件编译未移除 | 移除 `__CCE_AICORE__ == 220` 和 `__NPU_ARCH__ == 3003/3113` 保护 |
| `__simt_vf__` 未声明 | 缺少 SIMT 头文件或编译选项 | 确认 `op_host/CMakeLists.txt` 中 ascend950 分支有 `-cce-aicore-dcci-before-kernel-end=false` |
| `RegTensor` 未声明 | 缺少 MicroAPI 头文件或命名空间 | 检查 `using AscendC::MicroAPI::RegTensor` |
| `CastTrait` 模板错误 | CastTrait 参数不匹配 | 检查四要素：RegLayout/SatMode/MaskMergeMode/RoundMode |
| `ToFloat<>` static_assert 失败 | A5 上 `ToFloat` 仅支持 BF16/FP8/HiFloat8 等新类型， 见下方「ToFloat<> 修复详解」 |
| UB 越界运行时错误 | Tiling 未预留 SIMT DCache 40KB | 升级到 L4 |

### ToFloat<> 修复详解

**错误现象**：
```
kernel_scalar_convert.h: error: static assertion failed: ToFloat only support
bfloat16_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/fp4x2_e1m2_t/fp4x2_e2m1_t data type on current device!
```
调用链：`ToFloat<>(val)` → `Cast<>(bVal)` → `static_assert` 失败

**典型修复示例（ctc_loss_v3）**：
```cpp
// 报错：ToFloat 不接受 
logProbBlank = ToFloat(logProbBlankTensor.GetValue(0));

// 修复：数据实际为 BF16（移除 BF16 条件编译后类型不匹配），先重解释再 ToFloat
logProbFirstChar = ToFloat(logProbFirstTensor.template ReinterpretCast<bfloat16_t>().GetValue(0)); // 修复后代码
```

### bisheng shebang 修复脚本

```bash
sed -i '1i#!/bin/bash' build/gen_bisheng_dir/bisheng
```

### 检查点
- [ ] 编译通过，生成 run 包
- [ ] 如失败，错误已定位并修复
- [ ] **编译通过后必须进入阶段 4 精度验证，不得跳过**

## 阶段 4：精度验证

**前置条件**：阶段 3 编译通过，算子 run 包已生成并安装到 CANN 环境。

### 核心原则

1. **精度闭环**：迁移后的 950 算子**必须**通过精度验证才算迁移完成，仅编译通过不够
2. **aclnn 接口调用**：所有精度测试**必须**通过 `torch.ops.npu.<op_name>(...)` 调用算子（EXEC_NPU_CMD 标准方式构建的绑定），**禁止**使用 PyTorch 原生同名接口
3. **源码驱动用例**：测试用例基于阶段 1 的「算子源码分析摘要」生成，以被迁移源码为准
4. **跨架构对比**：950 NPU 输出 vs CPU 参考实现（PyTorch float32 计算），确保数值正确性
5. **全 dtype 覆盖**：算子支持的每种 dtype 都必须测试，用例总数 ≥ 30

### Step 4.0：环境确认（引用阶段 0）

**MUST 首先检查阶段 0 的环境信息是否已获取且完整。若环境信息缺失，MUST 立即暂停并执行阶段 0 的环境信息获取流程（参见阶段 0）。**

**环境信息确认完整后，所有 Shell 命令中使用阶段 0 记录的变量**：
- `source ${SET_ENV_CMD#source }` — 激活 CANN 环境
- `${PYTHON_PATH}` — 使用已确认的 Python 解释器
- `ASCEND_RT_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICE}` — 指定可用 NPU 设备

**MUST NOT 在未确认环境信息完整前，执行任何 NPU 测试操作。**

### Step 4.1：算子安装

编译通过后，将算子安装到 CANN 运行环境：

```bash
cd <build_output_dir>
./custom_opp_<target>_run.sh install
```

验证安装成功：
```python
python3 -c "
import torch
import torch_npu
# 检查算子是否注册到 torch.ops.npu 命名空间
op_name = '<op_name>'  # 替换为实际算子名
try:
    getattr(torch.ops.npu, op_name)
    print(f'torch.ops.npu.{op_name} is available')
except AttributeError:
    print(f'torch.ops.npu.{op_name} is NOT available, check installation')
"
```

### Step 4.1.1：构建 PyTorch 绑定（当 torch.ops.npu 中无算子时 MANDATORY）

**触发条件**：Step 4.1 验证发现 `torch.ops.npu.<op_name>` 不存在（即算子未注册到 `torch_npu` 内置绑定）。

**核心认知**：CANN OPP 格式安装的算子包（`vendors/` 目录）**不会**自动注册到 `torch.ops.npu`。需要额外构建 PyTorch C++ Extension `.so`，通过 `TORCH_LIBRARY_FRAGMENT(npu)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1)` 将算子的 aclnn 接口注册到 PyTorch。

**唯一标准方式**：使用 `torch_aclnn_helper.h` 中的 `EXEC_NPU_CMD` 宏。**禁止**使用手动 aclnn C API（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）、`import ascend_kernel`、Pybind 直接封装等方式。

**MUST 读取**：[`references/precision-testing/pytorch-binding-build-guide.md`](references/precision-testing/pytorch-binding-build-guide.md)

**参考源文件**：[`references/precision-testing/torch_aclnn_helper.h`](references/precision-testing/torch_aclnn_helper.h)（CANN 标准桥接头文件，**必须**复制到绑定项目源码目录）

#### 判定流程

**所有迁移算子都必须构建 PyTorch 绑定，无例外。**

```
算子迁移完成
  └─ 必须使用 EXEC_NPU_CMD 构建自定义 PyTorch 绑定
     ├─ 即使算子有 PyTorch 原生同名接口（如 torch.gather），也必须构建绑定
     ├─ 即使 torch.ops.npu 中已有该算子，也必须构建绑定（确保走 950 迁移后的 aclnn 路径）
     └─ 构建完成后通过 torch.ops.npu.<op_name>(...) 调用
```

**为什么禁止使用 PyTorch 原生同名接口**：

| 场景 | PyTorch 原生接口（禁止） | EXEC_NPU_CMD 绑定（必须） |
|------|------------------------|--------------------------|
| `torch.gather(x, dim, index)` | 走 `torch_npu` 内置分发，950 上可能走非迁移路径 | 走 `aclnnGather`，950 上走 `gather_elements_apt`（迁移后路径） |
| `torch.acosh(x)` | 走 `torch_npu` 内置分发，无法确认 950 调度路径 | 走 `aclnnAcosh`，确认 950 上走迁移后路径 |
| `torch.sigmoid(x)` | 同上 | 同上 |

**核心原因**：PyTorch 原生接口通过 `torch_npu` 内置分发机制调用 aclnn，950 上的调度路径由 `torch_npu` 版本决定，**不受迁移控制**。使用 EXEC_NPU_CMD 绑定可以确保：
1. 走迁移后的 950 专用 aclnn 路径（RegBase kernel）
2. 精度测试验证的是迁移后的算子行为，而非 `torch_npu` 内置行为
3. 绑定代码可审计、可追溯

#### 构建流程摘要（EXEC_NPU_CMD 标准方式）

1. **从算子源码提取接口信息**（MUST 读取 `op_host/op_api/aclnn_<op_name>.h` 和 `.cpp`）
2. **创建 `<op_name>/pytorch/` 目录**，包含 `.cpp` 绑定文件 + `CMakeLists.txt` + `torch_aclnn_helper.h` + 测试脚本
3. **`.cpp` 文件核心内容**（EXEC_NPU_CMD 标准方式）：
   - `#include "torch_aclnn_helper.h"` — 一个 include 即可，不需要 `#include <acl/acl.h>` 或 `#include <aclnn/acl_meta.h>`
   - `EXEC_NPU_CMD(aclnnXxx, ...)` — 一行宏调用，自动处理类型转换 + workspace + 执行 + 释放
   - `TORCH_LIBRARY_FRAGMENT(npu, m)` 注册算子签名（不是 `TORCH_LIBRARY(custom_ops, m)`）
   - `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 绑定到 NPU 设备
4. **CMakeLists.txt 关键配置**：
   - 纯 C++ 编译（`LANGUAGES CXX`），**禁止**使用 ASC 编译器
   - **不需要**链接 `libcust_opapi.so`（EXEC_NPU_CMD 运行时动态加载）
   - **必须**链接 `dl`（dlsym 需要）
   - 包含 CANN 头文件路径（`${CANN_INCLUDE}` 和 `${CANN_ACLNN_INCLUDE}`，从阶段 0 环境信息推导）
   - 包含 torch_npu 头文件路径（`${TORCH_NPU_PATH}/include`，从阶段 0 环境信息推导）
   - 添加 `-D_GLIBCXX_USE_CXX11_ABI=1`（与 torch_npu ABI 一致）
5. **构建命令**：
   ```bash
   source ${SET_ENV_CMD#source }
   export LD_LIBRARY_PATH=<vendors>/<vendor>/op_api/lib:$LD_LIBRARY_PATH
   cd <op_name>/pytorch && rm -rf build && mkdir build && cd build
   cmake .. && make -j$(nproc)
   ```
6. **验证构建产物**：
   - `ls build/libcustom_ops.so` — 确认 .so 文件存在
   - Python 可加载：`python3 -c "import torch; torch.ops.load_library('build/libcustom_ops.so'); print('OK')"`
   - 运行 Python 验证脚本确认算子可调用

#### 关键踩坑点

| 踩坑 | 说明 |
|------|------|
| `NPU arch is not supported!!!` | 使用了 ASC 编译器，改为纯 C++ 编译（`LANGUAGES CXX`），文件扩展名 `.cpp` |
| `Cannot determine link language` | 文件扩展名 `.asc` 但无 ASC 编译器，改为 `.cpp` |
| `aclnnXxx not in libopapi.so` | 运行时找不到 `libcust_opapi.so`，设置 `LD_LIBRARY_PATH` |
| aclnn 返回非 0 错误码 | 必须设置 `ASCEND_CUSTOM_OPP_PATH` 指向 vendors 目录 |
| NPU tensor 创建卡住 | NPU 设备被占用，设置 `ASCEND_RT_VISIBLE_DEVICES` |
| `undefined symbol: _ZN...` | C++ ABI 不匹配，添加 `-D_GLIBCXX_USE_CXX11_ABI=1` |
| `no member named 'is_npu'` | `is_npu()` 不是 PyTorch 标准 API，改用 `x.device().type() == c10::DeviceType::PrivateUse1` |

#### 精度测试中的调用方式

构建 PyTorch 绑定后，精度测试脚本中使用 `torch.ops.npu.<op_name>(...)`：

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
import torch, torch_npu
torch.ops.load_library("<path_to>/libcustom_ops.so")

out = torch.ops.npu.<op_name>(x, gamma, beta, quant_scale, num_groups=8, eps=1e-5, activate_silu=True)
```

### Step 4.2：生成精度测试用例

**MANDATORY**：基于阶段 1 的「算子源码分析摘要」生成测试用例。

#### 用例设计原则

1. **全 dtype 覆盖**：每个 shape / 边界值遍历算子支持的全部 dtype
2. **shape 由算子决定**：根据源码分析摘要中的维度约束选择 shape
3. **shape 不要过大**：单个用例元素数 ≤ 200K，避免测试时间过长
4. **用例总数 = (len(TEST_SHAPES) + len(BOUNDARY_VALUES)) × len(SUPPORTED_DTYPES) ≥ 30**
5. **边界值基于源码**：从源码分析摘要的「输入域约束」和「边界条件」章节提取

#### 用例组成

**Part A：常规 Shape 测试（TEST_SHAPES）**

| 维度 | 推荐 shape | 适用算子类型 |
|------|-----------|-------------|
| 1D | (128,), (1024,), (4096,), (8192,) | elementwise, reduction |
| 2D | (32, 512), (64, 768), (128, 1024) | elementwise, matmul, linear |
| 3D | (8, 16, 64), (4, 128, 256) | elementwise, attention |
| 4D | (4, 8, 32, 16), (2, 64, 32, 32) | conv2d, elementwise |

**Part B：边界值测试（BOUNDARY_VALUES）**

| 算子类型 | 推荐边界值 |
|----------|-----------|
| 有域限制（如 acosh x≥1） | 域边界值 + 临近值 + 典型值 + 大值 |
| 有域限制（如 log x>0） | 临近零 + 1.0 + 典型值 + 大值 |
| 无域限制（如 relu） | 0.0, 1.0, -1.0, 100.0, -100.0 |
| 量化类（FP8/INT8） | 0.0, ±1.0, ±最大值, ±最小非零值, Subnormal |

### Step 4.3：生成 aclnn 精度测试脚本

**MANDATORY**：测试脚本**必须**通过 `torch.ops.npu.<op_name>(...)` 调用算子（底层走 aclnn 两段式调用），不使用 `import ascend_kernel` 直接加载 `.so` 的方式。

#### aclnn 调用链路

```
Python 层:  torch.ops.npu.<op_name>(x, ...)
    ↓ (PyTorch C++ Extension 注册)
C++ 层:     EXEC_NPU_CMD(aclnnXxx, x, ..., result)
    ↓ (宏展开)
C 层:       aclnnXxxGetWorkspaceSize(...)  →  计算工作空间大小
            aclnnXxxExecute(...)            →  执行算子计算
```

**核心认知**：Python 层面**不存在** `torch.ops.aclnn.*` 命名空间。所有 NPU 算子调用在 Python 层走 `torch.ops.npu.<op_name>(...)` 路径，底层由 C++ 的 `EXEC_NPU_CMD(aclnnXxx, ...)` 宏完成 aclnn 两段式调用。

#### 精度测试脚本模板

**MUST** 先读取 `references/precision-testing/` 目录下的模板文件，替换占位符后生成测试脚本。

| 模板 | 路径 | 生成目标 |
|------|------|---------|
| pytest 测试 | `references/precision-testing/test_op_precision_aclnn_template.py` | `test_<op_name>_precision.py` |
| 报告生成器 | `references/precision-testing/run_precision_report_aclnn_template.py` | `run_<op_name>_precision_report.py` |
| 报告模板 | `references/precision-testing/precision_report_template.md` | `<op_name>_precision_report.md` |

**参考文档**（阶段 4 执行前 MUST 读取）：

| 文档 | 路径 | 用途 |
|------|------|------|
| 精度标准 | `references/precision-testing/OPS_PRECISION_STANDARDS.md` | MERE/MARE 阈值定义 |
| aclnn 接口指南 | `references/precision-testing/aclnn-interface-guide.md` | 从 torch.ops.npu 迁移到 aclnn |
| 源码逆向分析 | `references/precision-testing/source-code-reverse-analysis.md` | 无文档算子的用例生成策略 |

#### Python 测试脚本生成规范

##### 文件清单与命名规则

| 文件类型 | 命名规则 | 存放位置 | 是否必须 | 生成方式 |
|---------|---------|---------|---------|---------|
| pytest 测试脚本 | `test_<op_name>_precision.py` | 算子项目根目录 | 必须 | 从模板替换占位符 |
| 报告生成器脚本 | `run_<op_name>_precision_report.py` | 算子项目根目录 | 必须 | 从模板替换占位符 |
| 精度报告 JSON | `<op_name>_precision_report.json` | 算子项目根目录 | 自动生成 | 报告生成器运行后产出 |
| 精度报告 MD | `<op_name>_precision_report.md` | 算子项目根目录 | 自动生成 | 报告生成器运行后产出 |

##### 双文件分工

| 文件 | 职责 | 运行方式 | 输出 |
|------|------|---------|------|
| `test_xxx_precision.py` | pytest 精度测试，判断通过/失败 | `pytest test_xxx_precision.py -v` | PASS/FAIL + 错误信息 |
| `run_xxx_precision_report.py` | 报告生成器，逐 case 打印精度指标 | `python run_xxx_precision_report.py` | JSON + MD 报告 + 终端逐 case 输出 |

**执行顺序**：先运行报告生成器获取详细指标，再运行 pytest 确认通过/失败。

##### 脚本内容结构规范

**1. 文件头部 docstring 格式**：

```python
#!/usr/bin/env python3
"""
Precision evaluation for <op_name> operator on ascend950.
Uses <npu_call_interface> interface (底层走 aclnn 两段式调用).

算子功能: <一句话描述>
公式: <数学公式>
aclnn 接口: aclnn<OpName>GetWorkspaceSize(...)
Python 调用: <torch.ops.npu.xxx 或 torch.xxx>
"""
```

**2. 占位符替换规则**：

| 占位符 | 替换为 | 示例 |
|--------|-------|------|
| `{{OP_NAME}}` | 算子名 | `gather_elements_v2` |
| `{{NPU_CALL}}` | NPU 调用表达式 | `torch.ops.npu.gather_elements_v2(x, dim, index)` |
| `{{CPU_REF}}` | CPU 参考实现表达式 | `torch.gather(x.cpu().float(), dim, index.cpu().long()).to(dtype)` |
| `{{SUPPORTED_DTYPES}}` | dtype 列表 | `[torch.float32, torch.float16, torch.bfloat16, torch.int32]` |
| `{{INPUT_LOW}}` | 随机输入下界 | `1.0` |
| `{{INPUT_HIGH}}` | 随机输入上界 | `11.0` |
| `{{TEST_SHAPES}}` | shape 列表 | `[("2D", "8x16 dim=0", (8, 16)), ...]` |
| `{{BOUNDARY_VALUES}}` | 边界值列表 | `[("index=0", 0), ...]` |
| `{{QUANT_DTYPE}}` | 是否包含量化类型 | `True` / `False` |
| `{{DETERMINISM_SHAPE}}` | 确定性测试 shape | `(8, 16)` |
| `{{DETERMINISM_RUNS}}` | 确定性测试重复次数 | `3` |
| `{{TOTAL_CASES}}` | 总用例数（预计算） | `81` |
| `{{NUM_STAGES}}` | 测试阶段总数（含算子特有） | `5` |
| `{{EXTRA_TEST_CONFIGS}}` | 算子特有配置变量（可为空） | `NEGATIVE_DIM_TESTS = [...]` |
| `{{EXTRA_TEST_RUN_BLOCKS}}` | 算子特有测试运行代码（可为空） | `for desc, shape, dim in NEGATIVE_DIM_TESTS: ...` |
| `{{EXTRA_TEST_CLASSES}}` | 算子特有 pytest 测试类（可为空） | `class TestNegativeDim: ...` |

**3. 数据生成方式（MUST 在 CPU 生成再搬到 NPU）**：

```python
# ✅ 正确：CPU 生成 → .to(device) 搬运到 NPU
def _make_random(shape, dtype, device):
    x = torch.rand(shape, dtype=torch.float32) * (HIGH - LOW) + LOW
    return x.to(dtype=dtype, device=device)

# ❌ 错误：直接在 NPU 上生成（部分算子如 torch.arange 在 NPU 上不可用）
def _make_random(shape, dtype, device):
    x = torch.rand(shape, dtype=torch.float32, device=device) * (HIGH - LOW) + LOW
    return x.to(dtype)
```

**4. 实时进度输出（MUST 确保用户能看到实时进度）**：

```python
# ✅ 正确：使用 log() 函数统一输出，flush=True 确保管道/重定向时也实时可见
def log(msg):
    print(msg, flush=True)

# ✅ 正确：每个用例显示 当前/总数 进度
log(f"  [PASS] Case {case_id:02d}/{TOTAL_CASES} {cat}/{desc} | ...")

# ✅ 正确：每个测试阶段开始前打印阶段标题
log(f"\n--- [1/{NUM_STAGES}] Regular Shape Tests ({n_cases} cases) ---")

# ❌ 错误：使用普通 print，管道/重定向时输出被缓冲，长时间看不到任何内容
print(f"  [PASS] Case {case_id:02d} ...")

# ❌ 错误：没有任何进度指示，用户无法判断程序是否在运行
```

**6. NPU 调用方式选择**：

**所有迁移算子必须使用 EXEC_NPU_CMD 构建的 `torch.ops.npu.<op_name>(...)` 调用，禁止使用 PyTorch 原生同名接口。**

```
算子迁移完成
  └─ 必须使用 EXEC_NPU_CMD 构建自定义 PyTorch 绑定
     ├─ 禁止使用 PyTorch 原生同名接口（如 torch.gather, torch.acosh, torch.sigmoid 等）
     │   原因：torch_npu 内置分发路径不受迁移控制，950 上可能走非迁移路径
     ├─ 禁止使用 import ascend_kernel 或 torch.ops.ascend_kernel.*
     └─ 构建完成后通过 torch.ops.load_library() + torch.ops.npu.<op_name>(...) 调用
```

**重要：aclnn 内部多路径调度**。aclnn L2 接口内部通常是多路径调度器，根据芯片型号和输入特征动态选择不同的 L0 底层算子。**MUST 在测试文件注释中记录 950 上的实际调度路径**，不能假设算子名与 aclnn 接口名一一对应。

示例（gather_elements_v2）：
```
torch.ops.npu.gather_elements_v2(x, dim, index)  # EXEC_NPU_CMD 绑定
  → aclnnGatherGetWorkspaceSize(...)
    → CalGather() 调度
      → 910B/910_93: IfUseGatherElementsV2() → l0op::GatherElementsV2
      → 950 (RegBase): l0op::GatherElements (gather_elements_apt 内核)
      → MOE 场景: l0op::GatherV2
```

**MUST 通过阅读 aclnn L2 源码（op_host/op_api/aclnn_*.cpp）确认 950 上的实际调度路径，而非假设。**

**7. CPU 参考实现选择**：

| 场景 | CPU 参考实现 | 示例 |
|------|------------|------|
| PyTorch 有同名接口 | 直接使用 | `torch.acosh(x.cpu().float()).to(dtype)` |
| PyTorch 有近似接口 | 组合实现 | `torch.nn.functional.gelu(x.cpu().float()).to(dtype)` |
| 无 PyTorch 对应接口 | 手写 NumPy 实现 | 基于源码分析摘要中的数学公式手写 |

**8. 量化输出 clamp 规范（MANDATORY for 量化算子）**：

当算子输出为整型（INT8/UINT8/INT4 等）时，CPU 参考实现**MUST** 在 `round` 后、类型转换前加 `clamp`，确保与 NPU 算子行为一致。

**依据**：NPU 算子输出整型时天然截断到目标类型范围，CPU 参考实现必须对齐此行为，否则溢出区域结果不一致。

| 输出类型 | clamp 范围 | 代码 | 来源 |
|---------|-----------|------|------|
| INT8 | [-128, 127] | `.clamp(-128, 127).to(torch.int8)` | 算子文档 `out` 参数 dtype=INT8 |
| UINT8 | [0, 255] | `.clamp(0, 255).to(torch.uint8)` | 算子文档 `out` 参数 dtype=UINT8 |
| INT4（存储为 INT8） | [-8, 7] | `.clamp(-8, 7).to(torch.int8)` | 算子文档 `out` 参数 dtype=INT4 |
| FP4 E2M1（存储为 UINT8/INT8） | [-6.0, 6.0] | `.clamp(-6.0, 6.0)` 后按算子文档存储方式转换 | FP4 E2M1 最大正常值 6.0 |
| FP8 E4M3FN | [-448.0, 448.0] | `.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)` | IEEE FP8 E4M3 最大正常值 |
| FP8 E5M2 | [-57344.0, 57344.0] | `.clamp(-57344.0, 57344.0).to(torch.float8_e5m2)` | IEEE FP8 E5M2 最大正常值 |
| FP16 | [-65504.0, 65504.0] | `.clamp(-65504.0, 65504.0).to(torch.float16)` | IEEE FP16 最大正常值 |
| BF16 | [-3.389e38, 3.389e38] | `.clamp(-3.389e38, 3.389e38).to(torch.bfloat16)` | BF16 最大正常值 |

**MUST 从以下源码文件确认输出 dtype**：
- 算子文档（`docs/aclnn<OpName>.md`）中 `out` 参数的「数据类型」列
- `op_host/op_api/<op_name>.cpp` 中 `AllocTensor` 的输出 dtype 定义
- `TORCH_LIBRARY` 的 `m.def(...)` 中返回值类型签名

```python
# ✅ 正确：round 后 clamp 再转整型
cpu_out = torch.round(silu_out / quant_scale).clamp(-128, 127).to(torch.int8)

# ❌ 错误：不加 clamp，溢出时 PyTorch 行为未定义
cpu_out = torch.round(silu_out / quant_scale).to(torch.int8)
```

**9. 量化输出精度判定**：

量化输出（整型）**MUST** 使用 MaxAbsErr 而非 MERE/MARE 判定：

```python
def _compute_metrics(npu_out, cpu_ref):
    # 量化输出（整型）：用 MaxAbsErr
    if npu_out.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32):
        diff = (npu_out.cpu().to(torch.int32) - cpu_ref.to(torch.int32)).abs()
        max_abs = diff.max().item()
        mean_abs = diff.float().mean().item()
        return max_abs, mean_abs, 0.0, 0.0, 0.0  # MARE/MERE/CosSim 不适用

    # 浮点输出：用 MERE/MARE
    npu_f = npu_out.cpu().float()
    ref_f = cpu_ref.float()
    abs_err = (npu_f - ref_f).abs()
    ...
```

| 输出类型 | 判定标准 | 说明 |
|---------|---------|------|
| 整型输出（INT4/INT8/INT16 等） | MaxAbsErr ≤ 1 | 浮点输入→整型输出的量化场景 |
| 浮点输出（FP4/FP8/FP16/BF16 等） | MERE < Threshold 且 MARE < 10×Threshold | 按通用浮点精度标准 |

**完整精度判定矩阵**（官方标准）：

| 输入类型 \ 输出类型 | 整型输出（INT4/INT8/INT16 等） | 浮点输出（FP4/FP8/FP16/BF16 等） |
|:---|:---|:---|
| 整型输入（INT4/INT8 等） | N/A | 参考通用浮点精度标准 |
| 浮点输入（FP4/FP8/FP16/BF16 等） | **MaxAbsErr ≤ 1** | 参考通用浮点精度标准 |

**10. 输入数据 NaN 检查（MANDATORY）**：

精度测试脚本**MUST** 在调用算子前检查输入数据是否包含 NaN：

```python
# MUST: 在 npu_call 之前检查
assert not torch.isnan(x).any(), f"Input x contains NaN!"
if quant_scale is not None:
    assert not torch.isnan(quant_scale).any(), f"quant_scale contains NaN!"
```

当 MaxAbsErr 异常大（>10）时，**首先排查输入 NaN**，而非直接判定为算子 bug。

##### 使用方法

```bash
# Step 1: 环境配置（使用阶段 0 记录的环境变量）
source ${SET_ENV_CMD#source }
export ASCEND_RT_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICE}

# Step 2: 运行报告生成器（获取详细精度指标）
${PYTHON_PATH} run_<op_name>_precision_report.py

# Step 3: 运行 pytest（确认通过/失败）
${PYTHON_PATH} -m pytest test_<op_name>_precision.py -v --tb=short

# Step 4: 查看报告
cat <op_name>_precision_report.md
```

#### 精度测试脚本示例（参考模板文件）

> 完整模板见 `references/precision-testing/test_op_precision_aclnn_template.py`，以下为关键结构示意。

```python
"""
Precision evaluation for <OP_NAME> operator on ascend950.
Uses torch.ops.npu interface (底层走 aclnn 两段式调用).
"""

import torch
import torch_npu
import pytest
import numpy as np

SUPPORTED_DTYPES = [<从源码分析摘要提取>]

THRESHOLD = {
    torch.float32:       2**-13,    # ≈ 1.22e-4
    torch.float16:       2**-10,    # ≈ 9.77e-4
    torch.bfloat16:      2**-7,     # ≈ 7.81e-3
    torch.float8_e4m3fn: 2**-3,     # FP8 E4M3
    torch.float8_e5m2:   2**-2,     # FP8 E5M2
}

TEST_SHAPES = [<从源码分析摘要提取>]
BOUNDARY_VALUES = [<从源码分析摘要提取>]
BOUNDARY_SHAPE = (1024,)

@pytest.fixture(scope="module")
def device():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def npu_call(x, **kwargs):
    """通过 torch.ops.npu 调用算子 - 根据具体算子适配"""
    # 替换为实际调用，如: torch.ops.npu.acosh(x)
    return torch.ops.npu.<op_name>(x, **kwargs)


def cpu_reference(x, dtype, **kwargs):
    """CPU 参考实现 - 根据 PyTorch 标准库或手写实现"""
    # 替换为实际 CPU 参考实现
    # 示例: return torch.acosh(x.cpu().float()).to(dtype)
    pass


def _compute_metrics(npu_out, cpu_ref):
    """计算精度指标"""
    npu_f = npu_out.cpu().float()
    ref_f = cpu_ref.float()
    abs_err = (npu_f - ref_f).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = abs_err / (ref_f.abs() + 1e-7)
    mare = rel_err.max().item()
    mere = rel_err.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        npu_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)
    ).item()
    return max_abs, mean_abs, mare, mere, cos


class TestRegularShapes:
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("cat,desc,shape", TEST_SHAPES)
    def test_shape(self, device, cat, desc, shape, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = torch.rand(shape, dtype=torch.float32, device=device)
        x = x * (INPUT_HIGH - INPUT_LOW) + INPUT_LOW
        x = x.to(dtype)
        npu_result = npu_call(x)
        cpu_ref = cpu_reference(x, dtype)
        max_abs, mean_abs, mare, mere, cos = _compute_metrics(npu_result, cpu_ref)
        assert mere < thresh and mare < 10 * thresh, \
            f"[{cat}] {desc} dtype={dtype} MERE={mere:.2e} MARE={mare:.2e}"


class TestBoundaryValues:
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("desc,value", BOUNDARY_VALUES)
    def test_boundary(self, device, desc, value, dtype):
        thresh = THRESHOLD.get(dtype, 2**-10)
        x = torch.full(BOUNDARY_SHAPE, value, dtype=dtype, device=device)
        npu_result = npu_call(x)
        cpu_ref = cpu_reference(x, dtype)
        max_abs, mean_abs, mare, mere, cos = _compute_metrics(npu_result, cpu_ref)
        assert mere < thresh and mare < 10 * thresh, \
            f"[Boundary] {desc} dtype={dtype} MERE={mere:.2e} MARE={mare:.2e}"
```

#### PyTorch 绑定调用方式（唯一标准）

**所有精度测试脚本**必须使用 `torch.ops.npu.<op_name>(...)` 调用算子：

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import torch
import torch_npu

# 如果算子未注册到 torch.ops.npu，需先加载自定义绑定
torch.ops.load_library("<path_to>/libcustom_ops.so")

# 调用算子
out = torch.ops.npu.<op_name>(x, ...)
```

**关键认知**：
- `torch.ops.npu.<op>` 底层由 C++ 的 `EXEC_NPU_CMD(aclnn<Op>, ...)` 宏完成 aclnn 两段式调用
- Python 层面**不存在** `torch.ops.aclnn.*` 命名空间
- **禁止**使用 PyTorch 原生同名接口（如 `torch.gather`、`torch.acosh` 等），必须使用 EXEC_NPU_CMD 绑定
- **禁止**使用 `import ascend_kernel`（仅开发阶段可用，精度测试和生产环境禁止使用）
- **禁止**使用 `torch.ops.custom_ops.*`（必须注册到 `npu` 命名空间）
- **禁止**使用手动 aclnn C API 调用（必须使用 `EXEC_NPU_CMD`）

### Step 4.4：执行精度测试

```bash
source ${SET_ENV_CMD#source }
export ASCEND_RT_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICE}
${PYTHON_PATH} -m pytest test_<op_name>_precision.py -v --tb=short
```

### Step 4.5：精度标准

采用**生态算子开源精度标准**（MERE/MARE）：

**通过标准**：MERE < Threshold **且** MARE < 10 × Threshold

| dtype | Threshold | MERE 上限 | MARE 上限 (10×) |
|-------|-----------|----------|----------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| HiFLOAT32 | 2⁻¹¹ | 2⁻¹¹ | 10 × 2⁻¹¹ |
| FLOAT8 E4M3 | 2⁻³ | 2⁻³ | 10 × 2⁻³ |
| FLOAT8 E5M2 | 2⁻² | 2⁻² | 10 × 2⁻² |

**辅助指标**（用于分析，不作为判定依据）：

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| MaxAbsErr | `(npu - ref).abs().max()` | 最大绝对误差 |
| MeanAbsErr | `(npu - ref).abs().mean()` | 平均绝对误差 |
| CosineSim | `F.cosine_similarity(npu, ref)` | 余弦相似度（1.0 为完全一致） |

### Step 4.6：精度问题排查

**当精度测试失败时**，按以下优先级排查：

| 失败现象 | 最可能原因 | 排查方向 |
|----------|-----------|---------|
| FP16 失败，FP32 通过 | arch35/ 中未升精度到 FP32 计算 | 检查 arch35/ kernel 的 Cast 路径 |
| 输出全零 | CopyOut 未执行 / GM 偏移错 | 检查 arch35/ 的 DataCopy 逻辑 |
| 输出含 NaN/Inf | 除零 / 溢出 / BF16 条件编译移除后类型不匹配 | 检查 Compute 逻辑和 ToFloat 修复 |
| 全部偏差，CosSim≈1 | 系统性精度损失 | 检查升精度路径是否完整 |
| 仅尾部元素错 | 尾 tile 长度 / 对齐问题 | 检查 curTileLength 计算 |
| 周期性错误 | tile 边界 / 搬运偏移 | 检查 DataCopy 偏移计算 |
| L2 量化精度异常 | CastTrait SatMode 错误 / 三步量化缺失 | 检查量化路径的 SatMode 和中间类型 |
| L3 SIMT 结果不确定 | 多线程数据竞争 | 检查是否需要 AtomicAdd |

**深度排查**：参考 `ascendc-operator-precision-debug` skill 的五阶段流程（误差分析 → 代码审查 → 实验隔离 → 插桩定位 → 修复验证）。

### Step 4.7：生成精度报告

**MANDATORY**：生成精度报告并在对话中展示结果。

报告包含：
1. **总览表**：总用例/通过/失败/通过率
2. **精度阈值标准表**
3. **常规 Shape 测试结果表**（按 category 分组）
4. **边界值测试结果表**
5. **按 dtype 汇总统计**
6. **关键发现**（≥3 条结论）
7. **950 vs 910b 精度对比**（如有 910b 基线数据）

### 检查点
- [ ] 算子已安装到 CANN 运行环境
- [ ] 精度测试脚本使用 `torch.ops.npu.<op_name>(...)` 调用算子（非 `import ascend_kernel`）
- [ ] 用例数 = (shapes + boundary) × dtypes ≥ 30
- [ ] 算子支持的每种 dtype 都已测试
- [ ] pytest 精度测试全部通过
- [ ] 精度报告已生成
- [ ] **精度测试结果已以 Markdown 表格形式展示在聊天界面**
- [ ] 精度失败时已按排查流程定位并修复

## 阶段 5：结果评估

### 评估维度

| 维度 | L1 成功 | L3 成功 | L2 成功 | 需升级 |
|------|---------|---------|---------|--------|
| 编译 | ✅ 通过 | ✅ 通过 | ✅ 通过 | ❌ 失败 |
| 精度 | ✅ MERE/MARE 通过 | ✅ MERE/MARE 通过 | ✅ MERE/MARE 通过 | ❌ 精度不达标 |
| UB 合规 | 需确认 | 需确认 | 需确认 | 明确不合规 |
| 数据类型 | 可能扩展 | 无扩展 | 可能扩展 | - |
| 性能 | 基准 | Scatter/Gather 提升 | 显著提升 | - |

### 最终检查
- [ ] 编译成功
- [ ] **精度验证通过**（MERE/MARE 达标，所有 dtype ≥ 30 例测试通过）
- [ ] 迁移层级判定与实际结果一致
- [ ] 如需升级到 L4/L3/L2，已明确列出原因和下一步

## 绝对不要做

- ❌ 不做评估直接改造，跳过层级判定
- ❌ 忘记创建 `config/ascend950/` 目录（`binary.json` + `simplified_key.ini`）
- ❌ 在 910b 的 `aicore_config` 上直接修改数据类型（950 应独立配置）
- ❌ 忽略 `op_host/CMakeLists.txt` 中已有 ascend950 分支（直接复用）
- ❌ 对同一算子调用两次 `add_modules_sources`（会导致 `COMPILED_OP_DIRS` 重复，target 名冲突）
- ❌ `COMPUTE_UNIT` 与 `TILING_DIR` 列表长度不等（`find_value_by_key` 会 `FATAL_ERROR`）
- ❌ 用空字符串 `""` 作为 `TILING_DIR` 元素（CMake 会吞掉空字符串，导致列表长度不匹配）
- ❌ 950 与 910b 共用 tiling 时仍添加 `COMPUTE_UNIT`/`TILING_DIR` 参数（应使用隐式自动检测，仅添加 `config/ascend950/`）
- ❌ 编译失败时不落盘日志就猜测原因
- ❌ 忽略 `bisheng` 脚本 shebang 问题（`Exec format error` 的根因）
- ❌ 忘记移除 BF16 的 `__NPU_ARCH__ == 3003/3113` 保护
- ❌ 使用 `-soc=` 而不是 `--soc=`（单横线会报 usage 错误）
- ❌ L3 中 SIMT 核函数忘记标记 `__simt_vf__` + `LAUNCH_BOUND`
- ❌ L3 中 SIMT 函数内使用 `LocalTensor`/`RegTensor`（只能用 `__gm__` 指针和标量）
- ❌ L3 中忘记在 `_apt.cpp` 用 `__NPU_ARCH__ == 3510` 切换 SIMT/Memory-based
- ❌ L3 中多线程写同一地址不用 `Simt::AtomicAdd`（数据竞争）
- ❌ L2 中忘记用 `__VEC_SCOPE__` 包裹 MicroAPI 计算
- ❌ L2 中修改溢出模式后忘记恢复（`SetCtrlSpr` 后必须 `SetCtrlSpr` 恢复）
- ❌ L2 中 FP32→INT8 直接一步 Cast（910b 和 950 硬件都不支持，950 必须三步：FP32→INT16→FP16→INT8；910b 也是三步但用 INT32 作中间类型）
- ❌ L2 量化 CastTrait 用错 SatMode（量化用 SAT，反量化用 NO_SAT）
- ❌ L2 中 `ReduceSumCustom` 不替换为 `ReduceSum`（950 MicroAPI 版本）
- ❌ 使用编译期常量（如 `if constexpr`）替换 `if` 来规避报错分支（`if constexpr` 会在编译期消除分支，导致未选分支中的代码完全不被编译检查，可能隐藏真实的 API 不兼容或类型错误）
- ❌ 编译通过就认为迁移完成（**必须通过精度验证才算完成**）
- ❌ 精度测试使用 `import ascend_kernel` 直接加载 `.so` 的方式（**必须使用 `torch.ops.npu.<op_name>(...)`**）
- ❌ 精度测试使用 PyTorch 原生同名接口（如 `torch.gather`、`torch.acosh` 等）（**必须使用 EXEC_NPU_CMD 绑定构建的 `torch.ops.npu.<op_name>(...)`，确保走 950 迁移后路径**）
- ❌ 精度测试使用 `torch.ops.aclnn.*` 调用方式（**Python 层面不存在此命名空间**）
- ❌ 精度测试使用 `torch.ops.ascend_kernel.*` 调用方式（**禁止使用，必须注册到 `npu` 命名空间**）
- ❌ 精度测试使用 `torch.ops.custom_ops.*` 调用方式（**禁止使用，必须注册到 `npu` 命名空间**）
- ❌ 构建 PyTorch 绑定时使用手动 aclnn C API（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）（**必须使用 `EXEC_NPU_CMD`**）
- ❌ 构建 PyTorch 绑定时使用 `.asc` 文件扩展名（**必须使用 `.cpp`**）
- ❌ 构建 PyTorch 绑定时使用 ASC 编译器（**必须使用纯 C++ 编译**）
- ❌ 构建 PyTorch 绑定时使用 `TORCH_LIBRARY(custom_ops, m)` 注册（**必须使用 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`**）
- ❌ 构建 PyTorch 绑定时编译时链接 `libcust_opapi.so`（**EXEC_NPU_CMD 运行时动态加载，不需要编译时链接**）
- ❌ 构建 PyTorch 绑定时手动实现 `make_acl_tensor` / `dtype_to_acl` / `destroy_acl_tensor`（**必须使用 `torch_aclnn_helper.h` 的 `ConvertType` 体系**）
- ❌ 跳过算子源码分析摘要直接生成测试用例（**必须先完成源码分析**）
- ❌ 有文档时仅依赖文档生成测试用例而不分析源码（**测试用例以源码为准**）
- ❌ 精度测试未覆盖算子支持的全部 dtype（**每种 dtype 必须测试**）
- ❌ 精度测试用例数不足 30 例（**必须 ≥ 30 例**）
- ❌ 精度测试失败时不排查直接放宽阈值（**应先排查根因，确需放宽必须在报告中说明原因**）
- ❌ 仅生成精度报告文件而不在对话中展示结果（**必须在对话中展示总览和关键发现**）
- ❌ 精度报告缺少 950 vs 910b 对比（**如有基线数据必须对比**）
- ❌ 未确认环境信息就执行任何操作（**MUST 先完成阶段 0 的环境信息获取与验证**）
- ❌ 假设用户环境与默认路径一致（**不同用户的 CANN 路径和 Python 环境可能不同，必须通过阶段 0 获取**）
- ❌ 在 Shell 命令中写死 CANN 路径或 Python 路径（**必须使用阶段 0 记录的变量：`${SET_ENV_CMD}`、`${PYTHON_PATH}`、`${NPU_VISIBLE_DEVICE}` 等**）
- ❌ 假设 aclnn 接口名与底层 L0 算子名一一对应（**aclnn L2 是多路径调度器，950 可能走不同的 L0 算子，MUST 读源码确认**）
