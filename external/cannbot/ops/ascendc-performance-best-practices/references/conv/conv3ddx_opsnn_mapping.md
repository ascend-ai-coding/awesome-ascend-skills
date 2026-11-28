# ConvDx（反向卷积 / backprop_input）ops-nn → 直调映射参考

> 本文件供 Conv Step T 的 T-A（TTK 采集）和 T-B（ops-nn 研读）阶段使用，覆盖**反向卷积族**：
> `conv3d_backprop_input_v2`（Conv3ddx）、`conv2d_backprop_input_v2` 等 `*backprop_input*` / `*dx` 算子（arch35/ascend950）。
> 正向 Conv2D（conv2d_v2）的映射见 `conv2d_opsnn_mapping.md`，两者不通用：**反向卷积 M/N/K 定义、模板参数体系、tiling 字段完全不同**。
> 知识来源：ops-nn `conv/conv3d_backprop_input_v2` 源码级走读（tiling + kernel 全量分析）。

---

## 0. 反向卷积与正向卷积的本质差异（先读，决定一切设计）

反向卷积（deconv / transposed conv）dx = dedy ⊛ rot180(filter)，矩阵化为：

| 矩阵 | 数据 | M 维 | N 维 | K 维 |
|---|---|---|---|---|
| A | dedy（正向输出/反传输入） | **dedx_h×dedx_w（输入图平面）** | - | - |
| B | filter（**旋转180°+转置**） | - | **dedx_cin（输入通道）** | **Hk×Wk×Cout1**（dedy 通道） |
| C | dedx（反传结果） | hi×wi | cin | - |

与正向相比的关键差异：
1. **M 是输入图（大图），K 是输出通道**——正向恰好相反。所以反向的"L1 全载对象、baseM/baseN 最优值、切 K 阈值"全部不同。
2. **filter 需要旋转 180°**：由 `loadB2Condition` 控制（load2dv2 逆序 / Dn2NzTranspose 反排）实现，正向无此概念。
3. **backpropPad（反卷积等效 padding）**：`backpropPadUp/Down/Left/Right/Tail` 是"输出图相对输入图滑动窗口"的等效 pad，直接喂给 load3d 的 padList（上限 255）；正向 pad 语义不同。
4. **stride 展开放大**：`hoExpand=(ho-1)*strideH+1`、`woExpand` 同理，A 矩阵加载按放大后窗口走。
5. **kernel_d 是独立外层循环**，不算进 K 值（`kValue = Hk*Wk*Cout1*k0`）。
6. **stride>kernel 时存在跳过场景**（initOutputFlag 强制清零）；**stride>1 的子 kernel 拆分**是独立模板（kernel split，复杂度最高）。

## 1. 模板体系（tiling 侧，arch35）

注册优先级升序遍历（小号先命中）：

| 优先级 | 模板 | 准入要点 | kernel 类 |
|---|---|---|---|
| 96 | SmallKernel | **仅 SocVersionFuse**；D=1、group=1、dedy/y=NCDHW、filter=NDHWC/FZ；计算量阈值 | Conv3dDxSmallKernel（独立轻量实现） |
| 97 | KernelSplitFullLoad | stride>1 子 kernel 拆分 + B 全载 | Conv3dDxKsBlock |
| 98 | KernelSplit | stride>1 子 kernel 拆分 | Conv3dDxKsBlock |
| 99 | SmallShape | 不切 hkwk、kValue≤32768；期望轮次<2.5 且不均分（负载均衡主导） | Conv3dDxOswBlock |
| 100 | FullLoad | dk=1、M>N、wi≤512；MTE2 bound（m*n<(m+n)*512）或 FIXP bound（k≤128）；`kValue*bestBaseN*2B ≤ L1` | Conv3dDxOswBlock（enableFullLoad） |
| 101 | InnerProduct | **兜底，全场景** | Conv3dDxOswBlock |

**要点**：SmallShape/FullLoad/InnerProduct 共享同一 kernel 实现（数据驱动差异），直调抽取可合并为一个 host tiling 模块 + 一份 kernel。

### 1.1 inner product tiling 决策链（host 侧算法骨架）
```
GetShapeAttrsInfo:
  mValue = CeilAlign(hi*wi, 16)            # M 全长（输入图平面）
  nValue = CeilAlign(cin_g, 16)            # N 全长（enlarge 后每组 Cin）
  kValue = Hk*Wk*Cout1G*k0                 # K 全长（Dk 不算）
  GetLargeHkWkTilingMode:                  # L1 装不下最小 A+B 块时
    NO_TILING(0) / TILING_HK(1, 只装Wk) / TILING_HK_WK(2, 每次hkwk=1)
    后两者强制 initOutputFlag=1，baseM≤256（load3d pad 上限）
  CheckC04Enable: bf16+Cout≤4+stride/dilation=1 → c04（K 折叠 4Cout）
DoLibApiTiling:
  CalcKSegment: kValue ≥ 32768(不切)/8192(切hkwk)/2048(fp32) → splitK（Cout 轴核间切分）
  InitBaseMNK: kernelDHW>1 → (512,128,64/bBytes)；=1 → (256,256,128/bBytes)
               AdjustBaseMNK（L0 上界 + 尾块收缩 + HkWk 对齐的 baseK 搜索）
  SetSingleCoreInfo: M/N 分核（不切K）+ 负载均衡微调（N 收缩/ M 对齐 wi）
  CalStepK: EqualL1MatchStepMNK vs LadderMatchStepMNK 取优（stepKa/stepKb）
  IsL1ParamsValid 失败 → LegalProtection（ShrinkBaseK→ShrinkBaseMN→关 L1 DB 兜底）
  SetTilingCondition: enableVecTrans / loadB1Condition / loadB2Condition
```

### 1.2 baseK 的 HkWk 对齐约束（正确性核心）
L1→L0 的一个 K 装载块不能跨 HkWk 分形边界：`(stepKa*baseK) % lenHkWkC0 == 0 或 ≥ kValue`（IsHkWkAligned）。`CalculateOptimalBaseK` 从 maxBaseK 往下找能被 `kernel_w*k0 / kernel_h*k0 / lenHkWkC0` 整除的值。直调复刻 tiling 时**必须保留该对齐搜索**，否则切 Hk/Wk 场景 L0B 数据错乱。

## 2. tiling_key 解码（arch35）

`ASCENDC_TPL_ARGS_DECL(Conv3dBackPropInputV2Arch35, ...)` 共 5 个参数，**UINT LIST 参数按列表索引编码**（值恰好等于索引），按声明顺序从 bit0 逐字段移位：

| # | 字段 | 位宽 | bit位 | 取值 |
|---|------|------|-------|------|
| 1 | loadB2Condition | 8 | 0-7 | 0=NO_TRANSPOSE_NO_REVERSE, 1=TRANSPOSE_AND_REVERSE, 2=REVERSE_ONLY, 3=TRANSPOSE_ONLY |
| 2 | kernelSplitMode | 8 | 8-15 | 0=NO_SPLIT_KERNEL, 1=SPLIT_KERNEL_HW, 2=SPLIT_KERNEL_H |
| 3 | groupConvMode | 8 | 16-23 | 0=ORIGIN, 1=ENLARGE |
| 4 | isBasicBlockTiling | 1 | 24 | 0/1（arch35 恒 1） |
| 5 | loadB1Condition | 8 | 25-32 | 0=GM_TO_L1, 1=VEC_TO_L1_C04, 2=GM_TO_L1_NO_HK, 3=GM_TO_L1_NO_HK_WK, 4=SMALL_KERNEL |

```python
def decode_conv3ddx_tiling_key(key):
    return {
        "loadB2Condition": key & 0xFF,
        "kernelSplitMode": (key >> 8) & 0xFF,
        "groupConvMode": (key >> 16) & 0xFF,
        "isBasicBlockTiling": (key >> 24) & 0x1,
        "loadB1Condition": (key >> 25) & 0xFF,
    }
# 例：key=0x1000002 → loadB2=2(REVERSE_ONLY), split=0, group=0, basic=1, loadB1=0(全量)
```

### 2.1 loadB2Condition 语义（filter 旋转/转置策略）
- `TRANSPOSE_AND_REVERSE(1)`：最通用（NCDHW filter，DN(cout,cin,dk,hk,wk)→NZ(cout1,hkwk,cin1,cin0,cout0)，B2 加载时组内逆序）
- `REVERSE_ONLY(2)`：NDHWC 且 baseN≥HkWk / NCDHW 且 kernelDHW*cin*bBytes≤64B / fp32/hif8/enlarge/vecTrans（功能约束）
- `TRANSPOSE_ONLY(3)`：kernelDHW==1（无需逆序）
- `NO_TRANSPOSE_NO_REVERSE(0)`：fractal_z 直搬 / c04（AIV 已重排）

### 2.2 loadB1Condition 语义（B1 装载模式）
- `GM_TO_L1(0)`：全量 HkWk 装载
- `VEC_TO_L1_C04(1)`：c04 模式（AIV VGather 重排）
- `GM_TO_L1_NO_HK(2)`：切 Hk（B1 只装单 hk 行的 wk）
- `GM_TO_L1_NO_HK_WK(3)`：切 HkWk（每次只装 hkwk=1 的点）
- `SMALL_KERNEL(4)`：小 kernel 模板

## 3. tiling_data 字段（Conv3DBackpropInputArch35TilingData，约 90 字段）

TTK 采到 `bin_tiling_data` 后按 int32 LE 解码（u64 拆 lo/hi）。关键字段分组：

| 组 | 字段 | 说明 |
|---|---|---|
| 核调度 | coreNum、batchDim/groupDim/mDim/kDim/nDim/dDim | 实际用核数与多维分核（arch35 后者基本固定 1） |
| L0/L1 块 | baseM/baseK/baseN、stepKa/stepKb、al0/bl0/cl0/al1/bl1Pbuffer、iterateOrder | 基本块与 L1 K 段装载；pingpong=2 开 |
| 分形 | c0（=32B/dtypeBytes）、c0BitsA/B（log2） | fp16: c0=16 |
| shape | batch/cin/cout/cinG/coutG/cout1/cin1/cout1G/cin1G、dout/ho/wo/di/hi/wi、dk/hk/wk、group/oriGroup | cinG/coutG 为 enlarge 后每组通道 |
| 卷积属性 | strideD/H/W、padFront/Back/Up/Down/Left/Right、dilationD/H/W | 原始属性 |
| 反卷积专用 | backpropPadTail/Up/Down/Left/Right、khDilation、kwDilation、hoExpand、woExpand、dkHkWk、hkWk | load3d pad/窗口预计算 |
| 核间切分 | singleCoreBatch/Group/Cout/Cin/Din/M | inner product: singleCoreDin=1、singleCoreCout=coutG（c04=4） |
| 特性开关 | initOutputFlag、enableVecTrans、enableFullLoad、isBiasFullLoad、quantMode、hf32Flag、enlarge | |
| splitK | enableSplitK、useUbAccumForSplitK、kSegment、kSegmentTail、kValueSegment | Cout 轴分段 |
| kernel split | kSCoutFullLoad、kSUseWorkSpace | （本映射不展开） |
| 其他 | offsetX、fixedShiftVal、enRelu、singleIterateDk | |

## 4. ops-nn 文件索引（conv3d_backprop_input_v2，arch35 主线）

```
op_host/op_tiling/
  conv3d_backprop_input_v2_tiling.cpp                        # 入口（TilingRegistry 分发）
  conv3d_backprop_input_v2_tiling_data_arch35.h              # tilingData 结构（直调必须复制）
  arch35/conv3d_backprop_input_v2_inner_product_tiling.*     # 内积模板(101)：完整算法链
  arch35/conv3d_backprop_input_v2_fullLoad_tiling.cpp        # 全载模板(100)
  arch35/conv3d_backprop_input_v2_small_shape_tiling.cpp     # 小shape模板(99)
  arch35/conv3d_backprop_input_v2_small_kernel_tiling.cpp    # 小kernel模板(96, fuse)
  arch35/conv3d_backprop_input_v2_kernel_split*.cpp          # kernel split(97/98)
  arch35/conv3d_backprop_input_v2_common.h                   # L0/L1/Core/TilingRunInfo 参数结构
  common/conv_backprop_input_context_utils.*                 # runInfo 解析(SetRunInfoToV2)、backpropPad、enlarge(CalRealG)
op_kernel/
  conv3d_backprop_input_v2.cpp                               # kernel 入口（模板分发）
  conv3d_backprop_input_v2_arch35_tiling_key.h               # tilingKey 模板参数声明（直调必须复制）
  arch35/conv3d_backprop_input_v2_arch35.h                   # arch35 分发（→KsBlock/OswBlock/SmallKernel）
  arch35/conv3d_backprop_input_v2/
    conv3d_dx_rowc_block.h                                   # Conv3dDxOswBlock：外层调度（核任务分配/循环）
    conv3d_dx_block_base.h                                   # Conv3dDxBase：GM 偏移/stride
    conv3d_bp_input{,_config,_impl,_intf}.h                  # Intf/Impl 门面（CRTP 桥接到 Func）
    conv3d_dx_small_kernel.h                                 # SmallKernel 独立实现
    conv3d_backprop_input_v2_init_output_arch35.h            # initOutput 清零
    conv3d_backprop_input_v2_vec_transpose.h                 # AIV 前置转置
  arch35/convolution_3d_backprop/
    conv3d_bp_func_arch35.h                                  # 主计算流程（Init/Iterate/IterateAll/Compute）
    conv3d_bp_common_func.h                                  # ComputeForKIter（K 内层流水）
    conv3d_bp_large_attribute_func.h                         # 切Hk/HkWk（ComputeForTilingHkWk）
    conv3d_bp_impl_base.h / intf_base / config_base / register_arch35
    conv3d_bp_kernel_split_func.h / small_kernel_func        # kernel split / small kernel 函数
    impl/conv_bp_input_sub_func*.h                           # 搬运子函数全家桶（详见 §5）
examples/arch35/test_aclnn_conv3d_backprop_input_v2.cpp      # aclnn 基线参考
```

## 5. kernel 侧核心结构（T-B 必读摘要）

```
OswBlock.Process → CalBasicBlockCore（核任务循环）
  → IterateAll → [splitK? CalcSplitK_ : while(Iterate)]
  → Iterate：AdvanceIterIndices(m→n→din) + UpdateL1ComputeInfo（尾块/跳过/全载判定）
  → Compute → [loadB1==0 ? ComputeForNoTilingHWk : ComputeForTilingHkWk]
  → ComputeForKIter（K 内层）:
       for kIdx: LoadToA1/LoadToB1(GM→L1, dn2nz/nd2nz 直接进 L1 分形布局)
                 + L1 Preload(pingpong)
                 AIC: LoadToA2(load3d 滑窗 im2col+NZ) / LoadToB2(load2dv2 逆序/转置)
                      → Mmad(cmatrixInitVal 首块初始化) → L0C pingpong
  → GetTensorC → LoadL0c2Gm（Fixpipe NZ→DN/ND + quant/atomic）
```

指令与同步速查：
- A 侧：`DataCopy(Dn2Nz/Nd2Nz)` GM→L1（dedy ND→NZ 分形），`LoadData(load3d/LoadDataWithStride)` L1→L0A（fmatrix 滑窗，padList=backpropPad）
- B 侧：`DataCopy(Dn2Nz*)` GM→L1（filter 转置进 L1），`LoadData(load2dv2)` L1→L0B（逆序=rot180）
- 出：`Fixpipe(L0C→GM, COLUMN_MAJOR=NDHWC 输出 / ROW_MAJOR + quantPre=NDHWC)`；NCDHW 输出走 COLUMN_MAJOR（dstStride=diHiWi）
- 同步：`M_MTE1 / MTE1_M`（L0 装载↔MMAD）；AIC↔AIV flag 6/7（c04/enlarge B1 共写）、8（FIXP）、9（splitK cast）；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 混合核
- bias：GM→L1→BT(C2, 64B 对齐)，首个 Mmad 用 `Mmad(l0c, l0a, l0b, biasBT, params)` 初始化 C

## 6. 直调抽取方案（T-C 实施参考，已验证路线）

### 6.1 工程结构（历史 direct_invoke 验证过的构建方式）
```
direct_invoke/conv3ddx/
  op_kernel/    # 从 ops-nn 复制的 kernel 文件集（§4 的 op_kernel 部分，去掉 arch22）
  op_host/      # main.asc.cpp（kernel 入口）+ conv3ddx_tiling.h/.cpp（host tiling 简化版）
  CMakeLists.txt
  run_main.cpp  # x86 主程序：构造输入、算 tiling、<<<>>> 下发、event 计时、golden 校验
```
CMake 要点（bisheng 直调编译，已在 9.2.0 + dav-3510 验证）：
```cmake
set(CMAKE_CXX_COMPILER ...)  # 正常 host 编译器编 run_main
# kernel 目标：
add_custom_command(... $ENV{ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin/bisheng
  --npu-arch=dav-3510 -O3 <src>.asc.cpp -o <src>.o ...)
# host 侧链接 acl/aclrt；kernel 侧头文件 -I include (cann include)
```
布局/dtype 用宏开关（如 `CONV3DDX_DTYPE_FP16 / CONV3DDX_DX_LAYOUT_NCDHW`），映射到 `DTYPE_FILTER/DTYPE_OUT_BACKPROP/DTYPE_Y/FORMAT_*` 后实例化模板。

### 6.2 kernel 入口模板实例化
排除 kernel split 后需实例化的组合（fp16/NCDHW 主线）：
- loadB2 ∈ {0,1,2,3} × groupMode ∈ {0,1} × loadB1 ∈ {0,1,2,3}（共 32 个实例；实际按用例裁剪）
- 运行时分发：host 把选择值写进 tiling buffer 尾部附加字段（或独立 u32 GM buffer），kernel 入口 switch 到对应 `conv3d_backprop_input_v2<...>` 实例
- tiling 读取：`GET_TILING_DATA_WITH_STRUCT(Conv3DBackpropInputArch35TilingData, td, tiling)`（直调无需注册框架）
- 需包含 `inc/platform.h`、`inc/macro.h`（ops-nn common 头，路径加 -I）

### 6.3 host tiling 移植清单
1. 复制 `Conv3DBackpropInputArch35TilingData`（BEGIN_TILING_DATA_DEF 宏展开为普通 struct + SaveToBuffer/LoadRaw —— 直调下可用 memcpy 打包）。
2. 移植 runInfo 计算：`SetRunInfoToV2` 中的 shape/attr 解析 + **backpropPad 计算**（`CalBackpropPadBefore/After`：`(filter-1)*dilation - pad` 与 `input - output*stride + pad`）+ **enlarge(CalRealG)**（groups>1 时 N 分形对齐优化；直调首版可固定 enlarge=1 保守实现，group 用例走 ORIGIN）。
3. 移植 inner product 主链（§1.1）：GetLargeHkWkTilingMode → CheckC04Enable → CalcKSegment → InitBaseMNK/AdjustBaseMNK → SetSingleCoreInfo → CalStepK → LegalProtection → SetTilingCondition → SetTilingData。
4. 叠加 SmallShape（枚举 baseM/N）与 FullLoad（B 全载 + stepK 阶梯）准入分支——三模板合一。
5. 可裁剪：GetTilingFromRepo/tuning、IsSocVersionFuse 全家（fixedShift/quant/hif8/fp8/int8/a16w8）、打印函数、bias/scale（首版无 bias 用例）。
6. **必须保留**：HkWk 对齐搜索、L1 合法性兜底、dstStride(baseM/blockSize≤65535) 校验、`SetScheduleMode(1)` 等价物（直调独占核语义）。

### 6.4 workspace 与输出清零
- workspace 至少 16MB 对齐段（GetUserWorkspace 偏移）；splitK 非 fp32 需 `coreNum*CeilAlign(cin,n0)*CeilAlign(HW,m0)*4B` 累加区
- initOutputFlag=1 时先跑 `Conv3dDxInitOutput` 清零（或 host 侧 aclrtMemset——仅精度验证 run，perf run 禁止）
- 直调 `<<<blockDim, ublock, workspace, stream>>>`；blockDim=coreNum

### 6.5 baseline 与对标口径
- baseline：官方 aclnn `aclnnConvolutionBackward`（example 参考调用），同 shape 同输入
- 计时：acl event 或 msprof Task Duration；warmup 3 次 + perf 5 次取中位数；**禁止 perf run 前 memset/memcpy device 内存**
- 精度：CPU 参考实现（scipy/torch conv_transpose3d 或手写 6 层循环），fp16 atol/rtol=1e-3 量级


### 6.6 直调编译注意事项（CANN 9.2.0 + bisheng + dav-3510）

1. **公共设备头缺失**：kernel 链的 `inc/platform.h` 依赖 `kernel_utils.h`（`namespace ops { IsSame/Ceil/... }`）。
   完整可用集合从 CANN vendors 包取：`$CANN/vendors/<pkg>/op_impl/.../ascendc/inc/`（hashtable_common.h、
   kernel_safe_data_copy.h、kernel_utils.h、load_store_utils.h、macro.h、platform.h 六件套整体拷入工程 inc/）。
2. **ASCENDC_TPL 头的 printf 编译错**：`ascendc/host_api/tiling/template_argument.h` 在 host pass 调 `printf`，
   但 AscendC 命名空间内可见的只有 `__aicore__ printf`（来自 `using namespace __asc_aicore`）。修复：在 include
   kernel_operator.h 之后、tiling_key 头之前加 `#include <stdio.h>` + `namespace AscendC { using ::printf; }`。
3. **`__global__` 不能调 `__global__`**：直调入口须为唯一 `__global__`；把 ops-nn arch35 入口改为
   `__aicore__ inline`，入口内的 `KERNEL_TASK_TYPE_DEFAULT(...)` 必须移到自己的 `__global__` 函数首行
   （该宏仅允许出现在 kernel 函数内）。
4. **`__fp16` 转换内建缺失**：device 链接报 `__truncsfhf2/__extendhfsf2` undefined。共享代码（host+device 单 TU）
   中禁用 `__fp16` 强转，改手写 IEEE754 位级转换，并核对目标 dtype 的转换精度。
5. **ACL API 名称**：CANN 9.2.0 中事件 API 为 `aclrtRecordEvent`（无 `aclrtEventRecord`）；耗时用
   `aclrtEventElapsedTime`。
6. **tiling 读取**：`GET_TILING_DATA` 宏由 ASC 构建系统生成；直调需手写等价物——packed struct
   （`#pragma pack(1)`，字段顺序与 TILING_DATA_FIELD_DEF 声明一致，紧凑无对齐 padding）+ 设备侧 u32 循环拷贝
   宏。Conv3Ddx arch35 结构体共 336B（%8==0 与 PostTiling 校验一致）。
7. **filter 形状语义（高频正确性坑）**：反向卷积 filter dim1 是 **cin/groups（每组输入通道）**，不是真实 Cin；
   真实 Cin 取自 dedx(y) 形状 dim1。golden 索引、filterSize、dxCExtend(enlarge 计算)、CheckVecTrans 的 cin 判断
   都要用对维度。golden 的 filter 平铺 stride 用 filterCin。
8. **MIX_AIC_1_2 直调验证**：直调 `<<<N>>>` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 可正常
   launch（AIC 的 GetBlockIdx∈[0,N)，AIV 的 GetBlockIdx∈[0,2N) 且 `>>1` 得 AIC 号）。验证方法：各核写
   `out[blockIdx]` 后 D2H 检查。**注意**：若发现随机丢核，先查设备健康度（PCIe RAS 错误：`dl fc update
   timeout`/`dl mac retrain` 会使下发任务随机丢失、aclnn 首调 double-free 崩溃），不要急于怀疑代码。
9. **构建分工**：host tiling 工具独立按纯 C++ 编译，避免每次 Host 调整都重新实例化完整 Device 模板链。

### 6.7 直调调度与运行排查

1. **baseline 可用性**：对目标 shape 先运行官方入口，确认能够完成并得到正确输出；运行失败的路径不能作为性能 baseline。检查 host executor 生命周期，勿将 executor 复用错误当作设备故障。
2. **直调官方/注册 kernel 的 workspace 早退**：入口 `GetUserWorkspace(workSpace)` 依赖注册运行时设置的
   `g_sysWorkspaceReserved`；直调时若为空，可能整体早退、输出全零而同步正常返回。bootstrap kernel 中
   调用 `SetSysWorkspace` 的 block-local 状态不能跨 launch 持久，须核对目标入口的 workspace 契约。
3. **fatbin task-type 元数据**：`.ascend.meta.<kernel>` NOTE 的 TLV(tag=0x10) TYPE 中，KernelMetaType
   取值为 1=AIC_ONLY、4=MIX_AIC_1_1、5=MIX_AIC_1_2、6=AICORE。核对它与目标 kernel 类型、
   `.Mix_Kernel_Type_*` 符号及最终构建契约是否一致。不能仅根据源码中的属性或宏推断调度类型，也不要事后
   patch TYPE 绕过加载校验；用各核输出标记检查实际 block 覆盖。
4. **直调调度**：AIC/AIV 混合入口须使用匹配目标 SDK 的 MIX 声明和 launch 配置；核对模板裁剪后的实际类型、
   blockDim、context/stream 及 schedule mode。不能用单机产物的成功配方替代目标工程的调度验收。
5. **性能采集**：直调二进制用 `msprof --application="<cmd>" --output=<dir> -f`，取
   `mindstudio_profiler_output/op_summary_*.csv` 的 Task Duration(us)，同时核对 block 数。
   进程墙钟包含 `aclInit` 等初始化开销，不能用于 kernel 计时。

## 7. 与正向 Conv2D 映射的对照（迁移经验）

| 维度 | conv2d_v2（正向） | conv3d_backprop_input_v2（反向） |
|---|---|---|
| M 维 | ho*wo（输出图） | hi*wi（**输入图**） |
| K 维 | Cin1（输入通道） | HkWk*Cout1（**空间核×输出通道**） |
| N 维 | Cout | Cin |
| filter 处理 | 不旋转 | **rot180（loadB2 逆序）** |
| pad | 原 pad | **backpropPad 等效变换** |
| 独有特性 | inner batch/small channel/copy mode | c04/splitK/切Hk/切HkWk/kernel split |
| 共享 | —— | L0/L1 基本块方法论、pingpong、load3d/load2dv2/Fixpipe 指令族 |
