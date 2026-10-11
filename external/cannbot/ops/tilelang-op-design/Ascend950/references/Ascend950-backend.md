# Ascend950 后端技术参考

本文提供源码索引、平台约束和设计示例，按当前算子需要查阅。旧技能、示例注释或经验说明与所选版本的 API、后端实现不一致时，以当前实现为准；需求语义冲突则由主技能的输入处理流程澄清。

## 1. 源码定位与版本信息

本文以安装后的 `tilelang-op-orchestrator` 插件真实根目录为 `PLUGIN_DIR`，源码由 Ascend950 工作流统一准备。TK、TL 固定对应以下目录，不从当前工作目录或本技能目录推导，也不接受用户路径覆盖：

| 简称 | 仓库 | 安装后的源码根目录 | 工作流输出 |
|---|---|---|---|
| **TK** | ops-tilelang | `PLUGIN_DIR/repositories/Ascend950/ops-tilelang/` | `OPS_TILELANG_DIR` |
| **TL** | tilelang | `PLUGIN_DIR/repositories/Ascend950/tilelang/` | `TILELANG_DIR` |

路径引用仅限本仓库内的文档及上述克隆源码目录。TK、TL 是路径简称；记录来源时展开为安装后的实际路径，不引用其他工作区的仓库名称或绝对路径。下列索引依据所提供的源码内容；使用时核对安装版本是否包含对应文件和符号，缺失时报告具体缺口，不从其他仓库补齐。

- **TL 后端实现**：重点查阅 `TL/src/ascend/` 和 `TL/src/backend/`，即插件内的 `repositories/Ascend950/tilelang/src/ascend/` 和 `repositories/Ascend950/tilelang/src/backend/`。本文统一使用安装后的仓库名 `tilelang`；后文 Python API、示例和测试路径同样相对于 TL 根目录。
- **源码与运行环境**：源码目录存在不代表 Python 实际导入该版本；实际导入位置需单独核实。
- **版本信息**：生成报告时分别记录安装后 TK、TL 的实际 HEAD 及影响结论的本地改动，不沿用其他工作区的提交号，也不将索引核对视为算子测试通过。
- **架构配置**：`TL/tilelang/contrib/bisheng.py:get_npu_arch` 在未提供参数和环境变量时默认返回 `dav-3510`，相关测试见 `TL/testing/ascend/target/test_ascend_bisheng_arch.py`。该默认配置不等同于物理设备检测结果。
- **编译路径**：`ascend` 与 `pto` 共用 Ascend target kind，通过 target keys 选择不同 codegen。查阅 `TL/tilelang/ascend/target.py`、`TL/tilelang/ascend/codegen.py`、`TL/tilelang/ascend/execution_backend.py` 和 `TL/testing/ascend/target/test_tilelang_ascend_target.py`。使用工作流传入的目标后端，分别记录 target、codegen 与 execution backend。

使用 `rg --files` 定位文件，`rg -n` 查询函数或类，`git -C <root> rev-parse HEAD` 记录版本。路径失效时按符号重新定位。静态源码检索无需导入可能初始化设备的模块。

## 2. 源码与参考索引

以下每行标明源码根目录，行内路径均相对于该根目录。TK 的算子实现位于 `src/cann_ops_tilelang/`，参考实现和测试位于 `tests/`。引用示例前需核对其固定 shape、核数、dtype 和对齐条件，不直接泛化到当前 case。

| 设计主题 | 参考位置 |
|---|---|
| 算子接口与输入契约 | TK `src/cann_ops_tilelang/moe/normalize_weight.py`、`src/cann_ops_tilelang/engram/engram_gate.py`；核对 shape、dtype、连续性、空输入及返回值 |
| golden 与精度 | TK `tests/moe/normalize_weight_ref.py`、`tests/moe/test_normalize_weight.py`、`tests/testing/numeric.py`；追踪每个测试实际使用的比较函数和阈值 |
| 归约与逐行广播融合 | TK `src/cann_ops_tilelang/moe/normalize_weight_kernel.py:get_normalize_weight_kernel`；结合对应接口、参考实现和测试核对分母语义 |
| 归约与 Norm | TL `examples/ascend/example_rmsnorm.py`、`testing/ascend/language/test_tilelang_ascend_reduce.py`；核对示例固定核数与整除条件 |
| GEMM 与 Cube/Vector 数据传递 | TL `examples/ascend/example_gemm.py`、`examples/ascend/example_gemm_mixedkernel.py`、`tilelang/contrib/ptodsl/gemm.py`、`src/ascend/op/gemm.cc` |
| 内存分配、copy 与 dual_copy | TL `tilelang/language/copy_op.py`、`tilelang/ascend/language/allocate.py`、`tilelang/ascend/language/copy_op.py`、`src/ascend/op/copy.cc`、`testing/ascend/language/test_tilelang_ascend_copy_oob.py` |
| Kernel 与执行域 | TL `tilelang/language/kernel.py:Kernel`、`tilelang/ascend/language/kernel.py`（`Kernel`、`MixedKernel`）、`tilelang/ascend/language/frame.py`（`SimtVF`、`SimdVF`）、`tilelang/ascend/language/simd.py` |
| 自动调度、同步与内存规划 | TL `tilelang/ascend/pipeline.py`、`src/ascend/transform/`；查询 AutoSchedule、InsertSync、MergeUBAllocations 及所选后端实现 |
| Ascend 后端与公共实现 | TL `src/ascend/`（含 `op/`、`transform/`、`codegen/`）、`src/backend/`；AscendC/PTO codegen 分别查阅 `src/ascend/codegen/codegen_ascend.cc`、`src/ascend/codegen/codegen_pto.cc`，公共实现查阅 `src/backend/common/` |
| 核数配置 | TK `src/cann_ops_tilelang/config.py`：`get_num_ai_cores` 读取设备属性 `cube_core_num`；`get_num_vector_cores` 按其两倍计算，并允许 `OPS_TILELANG_NUM_VECTOR_CORES` 在检测值内设置上限。该关系是当前实现约定，设计时仍需核对目标设备与实际 task 数 |

性能参考位于 [本分支性能资料](../../../tilelang-performance-best-practices/Ascend950/references/index.md) 所在目录。从 `index.md` 选择相关算子族，并通过 `template_status.md` 核对验证状态。仅按需引用经验，不执行其完整测试和调优流程；未标明适用版本的策略仍需查证。

## 3. Ascend950 设计约束

1. **Kernel 与线程域**：依据 `TL/tilelang/ascend/language/kernel.py`，当前 Ascend 的 `T.Kernel(N)` 仅接受一维 block 网格，不接受 `threads=`。线程域由 `TL/tilelang/ascend/language/frame.py` 中的 `SimtVF` 定义；`MixedKernel` 的 `sid` 用于 AIV 子核分区。
2. **执行模式**：规则向量计算可参考 `SimdVF`，线程级索引、分支或归约可参考 `SimtVF`，矩阵乘法参考 Cube。两类 VF 的 API 和同步范围不同，CUDA warp/PTX 用法不能直接迁移。对当前所选操作核对对应 lowering。
3. **数据通路**：纯向量通常涉及 GM/UB/VF，GEMM 涉及 L1、L0A/B/C，融合方案可能使用 L0C→UB、UB→L1。依据 `TL/src/ascend/op/copy.cc` 与所选 codegen 确认合法通路，不采用所有数据必须逐层经过全部内存的假设。helper 隐式分配的 L0A/B 仍需计入资源模型。
4. **GEMM 布局与数值模式**：`TL/tilelang/contrib/ptodsl/gemm.py:PTOGemmL1Template` 要求 B 在 L1 中以 `[N,K]` 布局进入对应转置路径。逻辑输入 B 可为 `[K,N]`，但需设计有依据的搬运转换，保持数学语义。根据实际 lowering 核对转置、dtype、对齐、累加初始化及 HF32/量化模式；参数存在不代表所有组合受支持。
5. **dual_copy 分区与转换限制**：`TL/tilelang/ascend/language/copy_op.py:dual_copy` 依据源、目的区域或分配形状的 2:1 关系确定两个 AIV 的数据分区；它用于含 Cube 工作的混合内核。L0C→UB 双目的搬运不支持同时改变 dtype；需要 cast 时单独安排，并明确各 AIV 的输出范围。
6. **资源容量**：按存储层级与所属核分别核算物理布局、DMA/SIMD 访问范围、对齐、多版本和临时空间，各子核 UB 不合并预算。容量依据应来自目标平台资料、已有设备属性或对应版本实现，不复制 Ascend910 数值或从单个示例反推上限。容量未知时明确方案所需容量及核实依据。
7. **自动调度前提**：`TL/tilelang/ascend/pipeline.py` 由 `TL_ENABLE_AUTO_SCHEDULE` 控制 AutoSchedule、InsertSync 等 pass，MergeUBAllocations 另行执行；关闭自动调度不能视为同步问题自动解决。`Persistent`、`Pipelined` 与 buffer versions 组合应参考相容示例，并检查循环长度能否支撑流水级数。动态边界与依赖处理能力按实际实现判断。
8. **边界访问**：非整除部分按有效范围搬入搬出；计算读取完整 UB tile 时，明确剩余元素的填充或 mask。按指定 case 设计边界处理；整除 case 通过不代表尾块已验证。

设计报告仅引用当前算子相关的结论及来源。缺少依据的能力应说明成立条件与核实方法；静态源码分析不得描述为编译、精度或性能验证通过。

## 4. 设计深度示例

以下示例说明如何从源码提取设计结论，不规定通用 tile、流水深度或执行模式。示例仅依据源码分析，未新增设备测试。

### 4.1 MoE 权重归一化：归约、广播与尾块

TK `src/cann_ops_tilelang/moe/normalize_weight_kernel.py:get_normalize_weight_kernel` 将 token 按 128 行分块，通过 `Persistent` 分配任务。在 `SimtVF` 中沿 top-k 维归约，给分母加 `1e-20`，再逐行广播相除，输出分母与归一化权重。当前示例使用 FP32、128 个线程和四级 buffer versions，这些是实现参数，不是通用推荐值。

接口 TK `src/cann_ops_tilelang/moe/normalize_weight.py:normalize_weight` 要求输入是连续的二维 FP32 NPU Tensor，top-k 维必须为正；空 token 输入直接返回空输出。内核以 `valid_rows` 限制尾块搬入搬出，但 VF 内仍处理完整 tile，迁移时需分析无效行读取及其是否影响有效行，不能仅凭搬运切片宣称尾块已验证。

TK `tests/moe/normalize_weight_ref.py:normalize_weight` 的参考分母没有加 `1e-20`；`tests/moe/test_normalize_weight.py` 使用随机非负权重，并以 `atol=2e-6, rtol=0` 比较两个输出。设计报告应明确数值稳定项和零分母语义，不能将现有用例的比较方式当作所有输入的等价性证明。

### 4.2 归约：归约范围与中间状态

TL `examples/ascend/example_rmsnorm.py:rms_norm_fwd` 预先加载权重，逐行处理输入，在 `SimtVF` 中累加平方和、计算 rstd，最终写出结果与 rstd。可参考 `alloc_reducer`、初始化和 `finalize_reducer` 的组合，但固定核数及 `batch // N_CORES` 不能直接作为通用任务分配方法。

报告需明确 `rstd = rsqrt(sum(x*x) / D + eps)` 的除数为逻辑长度 D，而非 padding 后的 tile 长度。一行跨多个 tile 时，应说明部分平方和的累加位置、完整 rstd 的形成时机，以及输出阶段是否重新读取输入。分块或在线 Softmax 同样需给出部分最大值与指数和的合并方法，不得将各块独立归一化后直接拼接。

### 4.3 GEMM 融合：AIC/AIV 分工与输出分区

TL `examples/ascend/example_gemm_mixedkernel.py:gemm` 通过 `MixedKernel` 获取 `(bx, sid)`，在 K 循环中将 A/B 搬入 L1 并执行 GEMM，首次迭代以 `clear_accum=(kt == 0)` 初始化 L0C。随后通过 `dual_copy` 将输出 tile 分配给两个 AIV，各自写回对应半块。

按 M 二分时，完整输出 tile 为 `[BM, BN]`，每个 AIV 的 UB 保存 `[BM/2, BN]`。第 `sid` 个 AIV 的逻辑输出范围为：

```text
行：[m_tile * BM + sid * (BM/2), m_tile * BM + (sid+1) * (BM/2))
列：[n_tile * BN, n_tile * BN + BN)
```

该公式假设 BM 可二分且当前为完整 tile；尾块需额外裁剪有效范围。后处理需要 cast 时，依据 `dual_copy` 限制单独安排。采用 N 分区或其他方案时，应重新推导对应范围。

### 4.4 内存预算：物理分配与峰值占用

先确定每个 buffer 的物理布局和 padding，再计算单版本字节数及同时分配的版本数。低位打包类型按实际存储字节计算。每个存储层级取同时占用的最大值，并计入后端临时空间和必要余量。

假设每个 AIV 的 UB 存放两个输入和一个输出，均分配 4096 个 bf16 元素，且各有两个版本，则仅这三组 buffer 的占用为：

```text
UB_buffers = 3 × 4096 × 2 bytes × 2 versions = 49152 bytes = 48 KiB
```

该算式仅为 buffer 占用示例，不代表 Ascend950 容量或通用推荐参数。正式报告还需补充实际对齐空间、临时量和余量，并与有来源的容量比较。版本维度已包含在物理 shape 中时，不得重复乘版本数；采用共享存储时，需给出旧数据生命周期已结束的依据。
