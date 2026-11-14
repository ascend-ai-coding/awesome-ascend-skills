# Conv3DDW StreamK + 确定性计算模板

## 1. 模板用途

Conv3DDW 是 Conv3D **权重梯度**。本模板与 Conv3DDX KS 使用相同的五文件结构：运行期 Case、独立 Host tiling、完整 Kernel、入口、launch 声明。将文件写入工程即可调用 StreamK 与确定性计算，不依赖 ops-nn 注册、GE、runtime KB 或外部待实现计算函数。

完整轮由一个 AIC 独占输出块；不足半核的尾轮沿 `Batch×Dout` 或 `Ho` 拆分，各核写私有 workspace，两个 AIV 按固定树归约。源码推导见[流程](../conv3ddw_streamk_tuning_guide.md)。

## 2. 五文件结构

| 模板 | 工程文件 | 编译器 / 职责 |
|---|---|---|
| [conv3ddw_stream_k_tiling.template](conv3ddw_stream_k_tiling.template) | `op_kernel/conv3ddw_stream_k_tiling.h` | 公共 Case、输入/输出、Host API、容量公式 |
| [conv3ddw_stream_k_tiling_impl.template](conv3ddw_stream_k_tiling_impl.template) | `op_host/conv3ddw_stream_k_tiling.cpp` | g++：几何检查、基本块、L1、StreamK、workspace |
| [conv3ddw_stream_k_kernel.template](dav310/conv3ddw_stream_k_kernel.template) | `op_kernel/conv3ddw_stream_k_kernel.h` | bisheng：Cube 搬运/Mmad、MN 调度、固定树、输出重排 |
| [conv3ddw_stream_k_kernel_entry.template](dav310/conv3ddw_stream_k_kernel_entry.template) | `op_kernel/conv3ddw_stream_k_kernel.cpp` | bisheng：`__mix__(1,2)` + dtype 分派 + launch |
| [conv3ddw_stream_k_launch.template](dav310/conv3ddw_stream_k_launch.template) | `op_kernel/conv3ddw_stream_k_launch.h` | Host launch 声明 |

按表直接复制五个文件并保留文件中的版权声明。Host 与 Device 共用同一份 Case；两个目录均加入 include path。

## 3. 实现范围

| 项目 | 当前模板 |
|---|---|
| 架构 / SDK | Ascend950 / dav-3510；按 CANN 9.2 接口编译，Load3D 使用 WithStride 接口对 |
| 输入 | 连续 NCDHW X、dY，同 dtype：fp16 / bf16 / fp32 |
| 输出 | 连续 FP32 `[Cout,Cin,Kd,Kh,Kw]`，无 bias 梯度 |
| 卷积 | groups=1；支持 stride、dilation、六向非负 padding；输出维由 Host 推导 |
| 几何限制 | 原生 Load3D：Kh/Kw≤255，Sh/Sw≤63，Dh/Dw≤255，四向 HW pad≤255；Host 还检查 KStart、索引与容量 |
| 大规格 | 不拆 Wo、不拆 kernel、不扩维 group；容量无法满足则 tiling 返回 false |
| 核数 | `aicoreNum` 由调用方查询目标平台可用 AIC 数后提供；此模板使用每 AIC 配 2 个 AIV，启动前须检查实际 AIV 数足够 |
| 缓冲 | A1/B1、L0A/B 双缓冲；L0C 可单/双缓冲；Host 与 Device 资源静态断言一致 |
| 输出初始化 | Kernel 自带清零；调用方无需预先清零 dW 或 workspace |
| 确定性 | 同硬件、dtype/HF32、tiling、核数下重复执行；不同计划之间不保证逐位相同 |

`streamkMode` 为 AUTO / DISABLED / BATCH_DOUT / HWOUT。后两项指定候选拆分轴，仍受尾轮 `0 < T%P <= P/2`、实际 R≥2 和 L1 容量限制；返回的 `streamkType` 才是实际方案（0=MN、1=BD、2=Ho）。`c0Db` 为 0 自动 / 1 单缓冲 / 2 双缓冲。`hf32=1` 只允许 FP32。

## 4. Host 接入与构建

```cpp
#include "conv3ddw_stream_k_tiling.h"
#include "conv3ddw_stream_k_launch.h"

Conv3ddwSkTilingIn in{};
in.n=2; in.cin=3; in.cout=5;
in.di=4; in.hi=6; in.wi=7;
in.kd=2; in.kh=3; in.kw=3;
in.padUp=in.padDown=in.padLeft=in.padRight=1;
in.aicoreNum=8; // 确认当前设备有 8 AIC + 16 AIV 可用
in.dtype=ConvDwSkDType::FP16;
Conv3ddwSkTilingOut out{};
if (!Conv3ddwSkDoTiling(in,out)) { /* 报错或上层选择其他实现 */ }
auto c=ToCaseView(in,out);
// X=[2,3,4,6,7]，dY=[2,5,3,6,7]，dW=[5,3,2,3,3] FP32。
// 通过 aclrtMalloc 或框架 allocator 分配 c.workspaceBytes 字节；为 0 时传 nullptr。
int32_t status=launch_conv3ddw_sk(x,dy,dw,workspace,c,stream);
// 检查 status；异步提交后，让所有 tensor/workspace 保持存活直到 stream 完成。
```

此处调用形式与 Conv3DDX KS 一致，但 launch 内部必须使用 `aclrtLaunchKernelWithHostArgs` 配置 **schedule mode 1**。全核 Mode0 barrier 依赖该调度；不要替换成不带调度配置的 `<<<>>>`。Case 按值传入，无 tiling H2D copy、tiling key 或系统 workspace 前缀。用户 workspace 为 `2×usedCoreNum×256 KiB`，独立于输入输出，各并发调用分别分配。

CANN 9.2 的最小构建分工如下（`DW_SK_CANN` 指实际 SDK 根，`DW_SK_DIR` 指物化目录）：

```bash
export ASCEND_HOME_PATH="$DW_SK_CANN"
"$DW_SK_CANN/tools/bisheng_compiler/bin/bisheng" \
  -xasc --npu-arch=dav-3510 -std=c++17 -O2 -fPIC \
  -I"$DW_SK_CANN/asc/include" -I"$DW_SK_CANN/include" \
  -I"$DW_SK_DIR/op_kernel" -I"$DW_SK_DIR/op_host" \
  -c "$DW_SK_DIR/op_kernel/conv3ddw_stream_k_kernel.cpp" -o "$DW_SK_DIR/kernel.o"
g++ -std=c++17 -O2 -fPIC -I"$DW_SK_DIR/op_kernel" \
  -c "$DW_SK_DIR/op_host/conv3ddw_stream_k_tiling.cpp" -o "$DW_SK_DIR/tiling.o"
```

最终链接包含这两个对象及调用方代码。使用 SDK 的 ASC CMake target 可继承运行库；手动链接须包含 `ascendc_runtime ascendcl runtime register error_manager profapi ge_common_base unified_dlog mmpa dl ascend_dump c_sec`，并配置 SDK `lib64` 和实际驱动 `lib64/driver` 搜索路径。仅 `-lascendcl` 不足以链接 bisheng 生成的运行时桩。

## 5. 来源与适配边界

源仓：用户提供的 `ops-nn/conv/conv3d_backprop_filter_v2`，commit `6e68ee596a843731c15c56d314ff073e1d0629ff`，整理日期 2026-09-16。

Device 将 arch35 的计算依赖合并到一个头文件，`SOURCE:` 分节保存原路径、类名和函数名；最外层 `conv3ddw_sk` 隔离符号。原框架格式映射改为固定 CubeFormat，tiling 数据别名绑定 Case，Load3D 明确选择 CANN 9.2 WithStride 接口。Cube 数据流、尾轮同步、奇数 R 和单缓冲额外分片沿用源实现。

Host 重写为无 SDK/框架依赖的容量求解器，保留尾轮准入、BD/Ho 启发式、L1 回退、workspace 契约；不复刻 KB、Winograd、groupEnlarge 或原仓的所有基本块调优。因此数学语义与确定性机制可复用，不能宣称与原仓所有 shape 使用同一 tiling 或同等性能。

## 6. 验证

验证程序保留在：

- [conv3ddw_streamk_host_check.cpp](../../../tests/conv3ddw_streamk_host_check.cpp)：2,000 个 Host case，检查分核覆盖、容量和非法输入拒绝。
- [conv3ddw_streamk_npu_check.cpp](../../../tests/conv3ddw_streamk_npu_check.cpp)：146 个设备组合，对比 CPU golden，并检查三次运行逐位一致。

将五文件按 §2 写入工程后，Host 验证程序与 `conv3ddw_stream_k_tiling.cpp` 用 g++ 编译链接即可。设备验证程序按 §4 的构建方式链接 `kernel.o`、`tiling.o` 和 CANN 运行库，运行时以首个参数指定设备编号（默认 0）；需 Ascend950 且至少 8 AIC + 16 AIV。

归约控制、调度、设备精度、重复性和构建的验收要求见[流程文档 §9](../conv3ddw_streamk_tuning_guide.md#9-性能判断与验收要求)。性能需按目标 shape 测量。
