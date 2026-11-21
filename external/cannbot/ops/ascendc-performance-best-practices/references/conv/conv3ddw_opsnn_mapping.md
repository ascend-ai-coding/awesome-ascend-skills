# ConvDw（反向卷积 / backprop_filter）ops-nn → 直调映射参考

> 本文件供 Conv Step T 的 T-A（TTK 采集）和 T-B（ops-nn 研读）阶段使用，覆盖**反向权重梯度族**：
> `conv3d_backprop_filter_v2`（Conv3ddw）等 `*backprop_filter*` / `*dw` 算子（arch35/ascend950 主线，arch22 次之）。
> 反向输入梯度（ConvDx）见 `conv3ddx_opsnn_mapping.md`，正向 Conv2D 见 `conv2d_opsnn_mapping.md`，三者不通用：
> **dw 的 M/N/K 定义、滑窗所在侧、累加语义与 dx 完全不同**。
> 知识来源：ops-nn `conv/conv3d_backprop_filter_v2` 源码级走读（tiling + kernel 全量分析）。

---

## 0. dw 与 dx/正向卷积的本质差异（先读，决定一切设计）

dw（权重梯度）= Σ_{n,dout,ho,wo} x_window ⊙ dedy，矩阵化为：

| 矩阵 | 数据 | M 维 | N 维 | K 维 |
|---|---|---|---|---|
| A | dedy（正向输出/反传输入） | **Cout1G×C0（每组输出通道）** | - | **Ho×Wo（输出图平面）** |
| B | x（fmap，**load3d 滑窗 im2col**） | - | **Cin1G×C0×Hk×Wk（每组输入通道×空间核）** | Ho×Wo |
| C | y=dw（filter 梯度） | Cout | Cin×Hk×Wk | - |

与 dx 相比的关键差异：
1. **dw 是"小 M/N、大 K"规约问题**：M=Cout、N=Cin×HkWk 一般一块 L0C 就放得下（blockBaseM/N≤256 且 M*N*4B*2≤L0C 才开 dbL0C）；而 K 真实全长 = batch×Dout×Ho×Wo。因此**核间不常规切 K**（M/N/Dk/Group 分核），K 靠 L0C 内累加 + fixpipe atomic 分段写出。
2. **滑窗在 B 侧（fmap）**：dx 的 load3d 在 A 侧（dedy）、B 侧 filter 走 load2dv2 逆序；dw 恰好镜像——**B（fmap）用 load3d 滑窗 im2col，A（dedy）用 LoadData2D 直取**。直调复刻最易搞反。
3. **无 rot180 / 无 backpropPad**：dw 的 pad 就是原始 pads，直接喂 load3d padList（上限 255，超限走切 W/切 kernel）；C 矩阵本身就是 filter 布局（y= [Cout, Cin/g, Dk, Hk, Wk]）。
4. **输出是小张量 + 累加语义**：必须先 `Conv3dDwInitOutput` 清零 y，再 `enAtomic=1` 的 Fixpipe 累加；K 按 `SINGLE_BLOCK_ADD_THRESHOLD=65536`（256×256）分段控制浮点累加误差。
5. **尾轮 streamK + 确定性归约是常规路径**（不受 deterministic 开关控制）：尾块基本块再按 BatchDout 或 HWout 切分多核，AIC 算部分和写 workspace，AIV 固定顺序归约回 y。
6. **独有机制**：groupEnlarge（组通道扩维 + AIV 重排去填充）、splitWo（W 方向装载切分）、isSplitKernelHW（超大 kernel 逐 hkwk 点）、Winograd（fp32 3×3 stride1 专用模板）。

## 1. 模板体系（tiling 侧）

`REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", ...)`，优先级升序遍历（小号先命中）：

| 优先级 | 模板 | 准入要点 | kernel 类 |
|---|---|---|---|
| 0 | arch22 BasicBlock | **非 950**；确定性走 MN+StreamK（仅 NDC1HWC0）；否则 bf16 + group=1 + dilationD=1 + kernelHW∈{1,4,9,16} 且 M/N≥128 | Conv3dDwBasicBlockSplitMK/KN/MN/MNStreamK（arch22） |
| 1 | arch22 Base（tbe 兜底） | **非 950**；GetTbeTiling + 910B3/B2 固定映射表 + deterministic group_dim | Conv3dDw（tbe 风格，arch22） |
| 2 | **Winograd**（arch35） | **950**；NCDHW×3 + fp32 + 2D（D=dout=kd=1）+ 3×3 + stride/dilation=1 + group=1 + pad≤8 + 对称 pad + 有效 tile 率≥0.5 + (co/64)(ci/64)≥16 + 累加轴≤512000 | Conv2dDwWinograd |
| 3 | **StreamK**（arch35 主模板） | **950** + dtype∈{bf16,fp16,fp32}（hf32 可）；**兜底** | Conv3dDwBasicBlockMNStreamK / Conv3dDwBasicBlockStreamK |

**要点**：arch35 上 Winograd(2) 先于 StreamK(3) 命中；基本块 MN/MK/NK 绑核方向是 streamK 的子集（`Conv3DDWV2BasicBlockTilingArch35::IsCapable` 恒 false，由 StreamK 模板统一实现）。AIV 核数须 ≥2×AIC（CheckVectorCoreNum）。

### 1.1 streamK tiling 决策链（host 侧算法骨架，arch35）
```
GetShapeAttrsInfo:
  SetConv3dBpFilterV2RunInfo          # 三格式归一化 NCDHW；groups=ci/filter_c 推断；output_padding；SAME pad 重算
  CheckAttrs/CheckFormat/CheckKernelSize   # dtype 合法性；fmap/dedy 同格式；kd/kh/kw ≤ (D+pad-1)/dil+1
  CalcRealGroup: groups>1 且 filter=NCDHW → mag_factor 扩维（CanEnableGroupEnlarge：
      L0C / UB(含 stride=8 对齐的 useBaseN) / 基本块≤256 / L1 四重校验），失败回退 real_g=groups
  SetBasicBlockAttrsTiling:
      mValue = cout1G*C0（每组 Cout 元素数, enlarge 后）
      nValue = hk*wk*cin1G*C0（arch35 不含 kd —— Dk 是独立外层循环；arch22 group/dilD/deter 场景合并进 N）
      kValue = ho*wo
DoOpTiling (StreamK):
  GetTilingFromRepo → QueryBank 命中则直接翻译 tiling（tuning 知识库）
  InitSplitWOI:     # enableSplitW= fmap/dedy 同为 NCDHW 或同为 NDHWC 且 dtype∈{bf16,fp16,fp32}
      pad_l/r>255 → 阈值 256 否则 512；wo>阈值 → splitWo=128、splitWi=(splitWo-1)*strideW+kwDilW+1、tailWo/tailWi
  MultiCoreSplitMN: # 返回 false = tiling 失败（超 L1）
      iterateOrder = mValue > nValue ? 1 : 0
      checkLargeSpecs?（kh/kw/dilH/dilW>255 || strideH/W>63 || padH/W>255 || C0*hk*wk>65536）
        no  → tryNormalTiling:    InitBaseBlock + SetStepK4SplitMN
        yes → trySplitKernelHW:   isSplitKernelHW=true（L1 只装 1 个 hkwk16，splitWi=splitWo）
      仍超 L1 → trySplitWo:       ShrinkSplitWOIAndTryTiling（splitWo 从 128 按 2 的幂衰减到 k0）
      仍超 L1 → trySplitKernelAndWo（splitWo=k0=C0 + isSplitKernelHW，理论必成）
  AdjustSmallCaseBaseBlock:  # 小 case 调大 baseM/baseN 使 streamK 分满核（groupEnlarge 时禁止）
  DoStreamKTiling:
      totalCnt = mCnt*nCnt*kd*real_g（注意：不含 batch/dout —— 它们属于 K 累加）
      coreBindDirection = totalCnt < coreNum ? STREAM_K(1) : MN_STREAM_K(2)
      streamkCnt = totalCnt % coreNum；=0 / >coreNum/2 / 格式不支持 → NO_STREAMK_CALC(0)
      否则 coreStreamK = coreNum/streamkCnt，按"分核数多者→尾块比例大者→默认 BatchDout"选
      DoStreamkByBatchDout（STREAMK_BATCHDOUT=1）/ DoStreamkByHWout（STREAMK_HWOUT=2，只切 H 轴，
        stepK 递减校验 L1，失败回退 NO_STREAMK）
  DoLibApiTiling:  # 写 dwTiling：singleCoreM/N/K、baseM/K/N、stepKa/Kb、pbuffer、bound、splitWo、streamkType...
  PostTiling:      # memcpy tilingData（248B，%8==0）；SetBlockDim(usedCoreNum)；SetScheduleMode(1)
```
辅助决策：
- `InitBaseBlock`：baseM=均分 256 且 16 对齐；baseN 同理再**对齐到 hkwk*C0 的整数倍**（l1Cin1 = max(baseN/(hk*wk*16),1)）；M*N*4B*2≤L0C 才 dbL0C=2；baseK=GetBaseK（L0A/L0B 上限内，优先对齐 splitWo，16 对齐）。
- `SetStepK4SplitMN`：depthA1/depthB1 从 16 起、每次 -2 对称阶梯衰减试 L1（CalAL1Bound/CalBL1Bound，**对 splitWo 与 tailWo 取 max**），≤2 时关 dbL1 并从头再来；仍超则 ShrinkBaseBlock（先 K 后 MN 再回调 M/K）。
- `AdjustSmallCaseBaseBlockForStreamK/ForSingleCoreN`：枚举 mCnt/nCnt 调整 singleShapeM/N，要求 blockTotalCnt 整除 coreNum 且不超 L0C。

### 1.2 baseN 的 HkWk 对齐约束（正确性核心）
B 矩阵 N 轴 = cin×hkwk，L1 中 fmap 按 `[cin1 行 × hkwk]` 布局，load3d 一次沿 N 提取 baseN：
- **baseN 必须不跨 cin 分形边界**：`InitBaseBlock` 将 baseN 收缩到 `l1Cin1*hk*wk*C0` 的整数倍；hk*wk=9 时按 9*C0 对齐，其余按 C0。
- **B1 搬运行数**：`CalculateBl1Cin1CopyLen = Ceil(Ceil(baseN,16), hkwk)`，不足一行或非整除拖尾时 +1/+2 行（CalB1HiCopyParams 同逻辑）。
- 切 hkwk 场景（isSplitKernelHW）每次只算 1 个 hkwk 点：`mmad_.n = n0`、`load3d_.kExtension = n0`、l1W=1。
直调复刻 tiling 时**必须保留 baseN 的 hkwk 对齐与行数计算**，否则 L1B 数据错乱（对应 dx 的 baseK HkWk 对齐，但 dw 在 N 轴）。

## 2. tiling_key 解码（arch35）

`ASCENDC_TPL_ARGS_DECL(Conv3dBackPropFilterV2, ...)` 共 5 个参数，按声明顺序从 bit0 逐字段移位（bool=1bit，UINT=8bit；UINT LIST 按列表索引编码）：

| # | 字段 | 位宽 | bit位 | 取值 |
|---|------|------|-------|------|
| 1 | conv3DDWTemplateId | 8 | 0-7 | 1=TPL_STREAM_K（尾块 streamK），2=TPL_MN_STREAM_K（纯 MN 绑核） |
| 2 | isSplitKernelHW | 1 | 8 | 0/1（超大 kernel 逐 hkwk 点切分） |
| 3 | groupEnlarge | 1 | 9 | 0/1（组通道扩维） |
| 4 | winogradTilingFlag | 8 | 10-17 | 0=DISABLE, 1=tile1（fp32 H4W8）, 2=tile2（fp32 H2W16） |
| 5 | winogradResidentFlag | 1 | 18 | 0=FMAP 驻留, 1=DY 驻留 |

```python
def decode_conv3ddw_tiling_key(key):
    return {
        "conv3DDWTemplateId": key & 0xFF,        # 1=STREAM_K, 2=MN_STREAM_K
        "isSplitKernelHW": (key >> 8) & 0x1,
        "groupEnlarge": (key >> 9) & 0x1,
        "winogradTilingFlag": (key >> 10) & 0xFF,
        "winogradResidentFlag": (key >> 18) & 0x1,
    }
# 基本块：GET_TPL_TILING_KEY(coreBindDirection, isSplitKernelHW, groupEnlarge, 0, 0)
#   常见：0x000/0x100/0x200...（templateId=1/2 组合 8 种）
# winograd：GET_TPL_TILING_KEY(1, 0, 0, tilingFlag∈{1,2}, residentFlag∈{0,1})
#   即 key 含 winogradTilingFlag≠0 时 kernel 入口优先走 winograd 分支（templateId 位仍是 1）
```

**arch22 tiling_key**（`op_kernel/conv3d_backprop_filter_v2_tiling_key.h`）仅 1 个 UINT：conv3DDWTemplateId = key & 0xFF，取值 0=TPL_SPLIT_DFL（tbe）、1=M_K、2=K_N、3=M_N、4=M_N_STREAMK 绑核。
`ASCENDC_TPL_SEL` 合法组合全部绑定 `ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)`（AIC:AIV=1:2，AIV 用 GetSubBlockIdx() 区分）；入口处 `KERNEL_TASK_TYPE_DEFAULT` 在 opbuild 管线外是空操作（同 conv3ddx §6.7-3）。

### 2.1 关键字段语义
- `conv3DDWTemplateId`：host 侧由 `DoStreamKTiling` 的 coreBindDirection 决定（totalCnt<coreNum 且有尾块 → 1，否则 2）；二者共享同一 kernel 类，仅尾轮是否做 streamK 确定性计算不同。
- `isSplitKernelHW`：checkLargeSpecs 触发（kernel/dilation/stride/pad 超 load3d 限制或 C0*hk*wk>65536）；tiling 会打日志提示"可能报 AiCore timeout"。
- `groupEnlarge`：mag_factor>1 且 L0C/UB/基本块/L1 四重校验通过；kernel 侧 C 写出走 AIV 重排（VecPostProcess→Rearrange2Gm），**不兼容 AdjustSmallCaseBaseBlock**。
- `winogradTilingFlag/residentFlag`：SelectTemplate 按"空转 tile 块更少"选 tile1/tile2（同数时选 H4W16 内轴更大）；resident 按 ci/co 较小者驻留（ci>co → DY 驻留）。

## 3. tiling_data 字段（conv_bp_v2_kernel::TConv3DDwTiling，248B）

TTK 采到 `bin_tiling_data` 后按 int32 LE 解码（尾部 3 个 u64 拆 lo/hi）。关键字段分组：

| 组 | 字段 | 说明 |
|---|---|---|
| 核调度 | usedCoreNum（=blockDim）、streamkType、coreStreamK 隐含、singleCoreM/N/K、singleCoreBatchDout、singleCoreHo（=singleCoreK/wo）、singleCoreCout（=singleCoreM）、singleCoreCin（=l1Cin1*C0）、singleCoreGroup/Dk/Batch、splitWo | streamkType：0=无尾轮，1=尾块切 BatchDout，2=尾块切 HWout |
| L0/L1 块 | baseM/baseK/baseN、stepKa/stepKb、al0/bl0/cl0/al1/bl1Pbuffer、iterateOrder、al1Bound/bl1Bound | baseN 已 hkwk 对齐；bound 为 L1 单次装载上限（splitWo 尾块取 max） |
| 分形 | channelSize（=k0）、m0/n0（恒 16） | fp32: k0=8；HIF8: k0=32（bl0Pbuffer 可能强制 1） |
| shape | batch/cin/cout/cin1G/cout1G/dout/ho/wo/di/hi/wi/dk/hk/wk/group/realGroup | **cin1G/cout1G 是 enlarge 后每组通道元素数（非 C0 分形数，与 dx 语义不同）**；tailCinG=cin-cin1G*(realGroup-1) 在 kernel 侧算 |
| 卷积属性 | strideD/H/W、padFront/Back/Up/Down/Left/Right、dilationD/H/W、hf32Flag | 原始属性（无 backpropPad） |

arch22 的 `Conv3DBackpropFilterV2TilingData` = TConv3DDwTiling + params（mDim/nDim/kDim/batchDim/groupDim/dkDim 等）+ TConv3DDwBasicBlockTiling（singleCoreM/N/K、streamKType、coreStreamK、usedCoreNum），结构不同，解码需另行处理。

## 4. ops-nn 文件索引（conv3d_backprop_filter_v2）

```
op_host/
  conv3d_backprop_filter_v2_def.cpp                        # OpDef（950: x/out_backprop NCDHW|NDHWC, y NCDHW|NDHWC|DHWCN, 含 HIF8）
  conv3d_backprop_filter_v2_infershape.cpp                 # InferShape（filter_size 定 y；from_depthwise 特判改 shape）
  op_tiling/
    conv3d_backprop_filter_v2_tiling.cpp                   # 入口（IMPL_OP_OPTILING + TilingParse 平台信息）
    arch22/conv3d_backprop_filter_v2_base_tiling.*         # tbe 兜底(优先级1)：GetTbeTiling + B3/B2 固定映射表
    arch22/conv3d_backprop_filter_v2_basic_block_tiling.*  # arch22 基本块(优先级0)：MK/KN/MN/MN+StreamK 绑核
    arch35/conv3d_backprop_filter_v2_basic_block_tiling_arch35.*  # arch35 基类：runInfo/enlarge/L1 bound/shrink
    arch35/conv3d_backprop_filter_v2_stream_k_tiling.*     # arch35 主模板(优先级3)：完整决策链(§1.1)
    arch35/conv3d_backprop_filter_v2_winograd_tiling.*     # winograd(优先级2)：准入+tile 选择+workspace
    arch35/conv3d_backprop_filter_v2_common.h              # 常量（512/256/128 阈值、DB、LOAD3D_PAD_THRESHOLD）
    common/conv_backprop_filter_context_utils.*            # SetConv3dBpFilterV2RunInfo(shape/attr/output_padding/SAME/校验)
    common/conv_backprop_filter_tuning_tiling.*            # QueryBank 知识库 tiling（tuningtiling 注册）
  config/ascend950/conv3d_backprop_filter_v2_binary.json   # 预编译二进制清单（省在线编译）
op_kernel/
  conv3d_backprop_filter_v2.cpp                            # kernel 入口（__CCE_AICORE__==310→arch35，否则 arch22）
  conv3d_backprop_filter_v2_tiling_key.h                   # arch22 tilingKey（TemplateId 0-4）
  arch35/conv3d_backprop_filter_v2_arch35.h                # arch35 入口（winograd/StreamK/MNStreamK 分发 + workspace 早退检查）
  arch35/conv3d_backprop_filter_v2/
    conv3d_backprop_filter_v2_tiling_key.h                 # arch35 tilingKey 5 参数（直调必须复制）
    conv3d_backprop_filter_v2_tiling_data.h                # TConv3DDwTiling（直调必须复制）
    conv3d_backprop_filter_v2.h                            # Conv3dDw 基类（CalcBlockOffsetA/B/C 三格式 GM 偏移 + pad 跳过）
    conv3d_dw_v2_basic_block_arch35.h                      # MNStreamK/StreamK：外层调度（主轮/尾轮/确定性同步）
    conv2d_dw_winograd.h                                   # Conv2dDwWinograd 门面（transdata→正变换→mmad→逆变换）
    conv3d_backprop_filter_v2_init_output_arch35.h         # 输出清零（AIV Fill + flag 9）
    conv3d_bp_filter{,_config,_impl,_intf}.h               # REGISTER_DW_IMPL CRTP 门面
  arch35/conv3d_backprop/
    conv_bp_func_arch35.h                                  # 主计算流程（Init/Iterate/IterateAll/Compute/GetTensorC/VecPostProcess）
    conv_bp_func_common.h                                  # ComputeLoop（K 内层流水）+ splitW 参数更新 + ClearBaseMNL0C
    conv_bp_large_kernel_func.h                            # ComputeSplitKernelHW（超大 kernel 逐 hkwk 点）
    conv_bp_util_arch35.h                                  # 位运算/CealHkWk/CRTP 宏（DECLARE_IMPL/CHECK_FUN）
    conv_bp_{config,impl,intf}_base*.h / register_arch35.h # CRTP 基座（ContextData 全部运行时状态）
    impl/conv_bp_filter_sub_func.h                         # Init/Step 参数/UpdateMNIdx/MmadLocal
    impl/conv_bp_filter_sub_func_load_{a1,b1,common,gm_to_l1,l1_to_l0}.h   # A1(dedy)/B1(fmap) GM→L1→L0 搬运
    impl/conv_bp_filter_sub_func_store_l0c{,_dispatch}.h   # L0C→GM 出口（三格式 × DkHkWkEqOne/BaseNUndivided/Normal）
    impl/conv_bp_sub_func_deterministic_{calculate,common}.h  # streamK 确定性归约（UB add tree + Rearrange2Gm）
    winograd/*.h                                           # winograd 全家桶（transdata/transform/mmad/inv_transform/queue）
  arch22/…                                                 # arch22 kernel（tbe 路径 + 基本块 split + deterministic）
examples/arch35/test_aclnn_conv3d_backprop_filter_v2.cpp   # aclnn 基线参考（aclnnConvolutionBackward）
tests/st/arch35/ttk_kernel_conv3d_backprop_filter_v2.csv   # TTK 用例（含 group/dilation/hif32 等覆盖）
```

## 5. kernel 侧核心结构（T-B 必读摘要）

```
Conv3dDwBasicBlockMNStreamK.Process →（block_idx ≥ usedCoreNum 直接返回）
  CalBasicBlock：主轮 calRound_=Ceil(totalCnt, coreNum)，block_idx 步进 usedCoreNum
    → basicBlockIdx 分解 dkIdx/nCoreIndx/mCoreIndx/groupIdx
    → CalSingleBasicBlock：
        CalcReduceSegments（K 分段：Ceil(65536,wo) 限制 batchDout×ho 段；splitKernelHW 时 H 段还 ≤256）
        双层循环(batchDout 段 × ho 段)：
          CalcOffset（A/B/C 三格式 GM 偏移，group 用扩维后的 cout1G/cin1G）
          ReCalDkCinSingleCoreShape（dout 全在 pad 外→跳过；group 尾块 cin/cout 收缩）
          → dw_.SetOutBackprop/SetFmap/SetStart_idx/SetSingleShape → dw_.IterateAll(y, enAtomic=1)
  尾轮 streamkType≠0 → CalcStreamK：
      尾块再切 coreStreamK 份（BatchDout 或 HWout），SetDeterAddCore4StreamK 定 workspace 偏移
      AIC: CubeWaitVector → CalcIterateCube（reduceSegments 细分）→ GetTensorC(ws, 1, enSequentialWrite=true)
      AIV: ClearWorkspace（本核 ws 段清零）→ Iterate → DeterministicReduceKInUb（固定顺序归约→y）
      每 14 次跳过触发 PipeBarrier 防 CrossCoreSetFlag 计数溢出
dw_.IterateAll → while(Iterate): UpdateMNIdx（M/N 块走位 + L1 驻留/释放决策）
  → Compute → [isSplitKernelHW ? ComputeSplitKernelHW(hkwk 逐点外层, L0C 先 ClearBaseMNL0C 清零)
                                    : ComputeNormal]
  → ComputeNormal: batchDout 循环 → splitWo 循环 → ComputeLoop（K 内层）:
       ComputeLoadToB1/LoadToA1（GM→L1, Dn2Nz/Nd2Nz 直接进 L1 分形布局 + MTE2 预取 pingpong）
       AIC: ExecuteMTE1L0b（LoadL12L0b: load3d 滑窗 im2col L1→L0B, mStartPt 随 k 走位, kExtension=baseN）
            ExecuteMTE1L0a（LoadL12L0a: LoadData2D L1→L0A, dedy 直取无滑窗）
            → Mmad（cmatrixInitVal=首块）→ L0C pingpong
  → GetTensorC → LoadL0c2Gm（enAtomic=1 → SetAtomicAdd<float>）
       groupEnlarge: LoadL0c2UbForGroup + AIV VecPostProcess(Rearrange2Gm 重排去扩维填充)
       否则按 y 格式：NCDHW 直搬 | NDHWC Nz2Nd | DHWCN Nz2DHWCN，再按 dhwK==1 / baseN 整除 hkwk 与否细分
```

指令与同步速查：
- A 侧（dedy）：`DataCopy(Dn2Nz/Nd2Nz)` GM→L1（按 ho 行拷、M=cout1），`LoadData(LoadData2D)` L1→L0A；baseUseM==1 有专用整块路径；fp32 kStep=2
- B 侧（fmap）：`DataCopy(Dn2Nz/Nd2Nz)` GM→L1（按 CalB1HiCopyParams 行数拷，pad 上界裁剪），`LoadData(load3d V2)` L1→L0B（**滑窗在 B 侧**，padList=[左,右,padUp,255]，l1H/l1W，filterH/W=1）
- 出：`Fixpipe(L0C→GM)`；首轮结果依赖 initOutput 清零 + atomic 累加；workspace 顺序写用于确定性归约
- 同步：`M_MTE1 / MTE1_M`（L0 装载↔MMAD）；initOutput AIV→AIC flag 9（SYNC_MODE2）；groupEnlarge AIC↔AIV flag 5/6（SYNC_MODE4）；streamK 确定性 flag 2/4/6/8（MODE0/MODE2）；AIC:AIV=1:2（MIX_AIC_1_2，AIV `GetSubBlockIdx()>0` 的子块跳过部分工作）
- workspace（arch35）：streamkType≠0 时 `2 × usedCoreNum × l0c_size`（Cube 顺序写 + Vector 归约各一份）；winograd 另有驻留区+transdata 区+尾块切 k 区；入口 `SetSysWorkspace/GetUserWorkspace` 为空且需要 ws → **整体早退、输出全零且 sync 正常**（易误判精度 bug，同 conv3ddx §6.7-2）

## 6. 直调抽取方案（T-C 实施参考）

### 6.1 工程结构（沿用 conv3ddx §6.1 验证过的构建方式）
```
direct_invoke/conv3ddw/
  op_kernel/    # 从 ops-nn 复制 arch35 kernel 文件集（§4 的 op_kernel/arch35 部分 + 根入口改造）
  op_host/      # main.asc.cpp（唯一 __global__）+ conv3ddw_tiling.h/.cpp（host tiling 简化版）
  CMakeLists.txt  # bisheng --npu-arch=dav-3510 -O3；host 链接 acl/aclrt
  run_main.cpp # 构造输入、算 tiling、<<<blockDim=usedCoreNum>>> 下发、event 计时、golden 校验
```
通用编译坑位（公共设备头六件套、ASCENDC_TPL printf、`__global__` 不能调 `__global__`、`__fp16` 内建、aclrtRecordEvent、tiling 手写 packed struct + u32 拷贝、task-type 元数据/调度残缺、msprof 采集口径）**直接沿用 `conv3ddx_opsnn_mapping.md` §6.6/§6.7**，不再重复。

### 6.2 kernel 入口模板实例化
- arch35 主线（fp16/NCDHW）：templateId{1,2} × isSplitKernelHW{0,1} × groupEnlarge{0,1} × winogradFlag{0}，共 **8 个实例**；winograd 用例再叠加 winogradTilingFlag{1,2} × residentFlag{0,1}（仅 fp32）
- 运行时分发：host 把选择值写 tiling buffer 尾部附加字段（或独立 u32 GM buffer），入口 switch 到对应 `conv3d_backprop_filter_v2<...>` 实例；**入口的 workspace 早退检查必须保留**（streamkType≠0 / winograd 时 user1 为空 → return）
- tiling 读取：`GET_TILING_DATA_WITH_STRUCT(conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData, td, tiling)` 等价手写（248B packed，%8==0）

### 6.3 host tiling 移植清单
1. 复制 `TConv3DDwTiling`（plain struct，memcpy 打包即可）。
2. 移植 `SetConv3dBpFilterV2RunInfo`：NCDHW/NDHWC/DHWCN 三格式归一化、groups=ci/filter_c 推断与整除校验、output_padding（仅 950 非零）、SAME padding 重算（ReCalPaddings）、**CheckGradOutputShape 的 ho/wo 期望值校验公式（x 与 out_backprop 角色互换）**——golden 与 tiling 都依赖该口径。
3. 移植主链（§1.1）：CalcRealGroup（enlarge 判定 + CanEnableGroupEnlarge 四重校验）→ SetBasicBlockAttrsTiling → InitSplitWOI → MultiCoreSplitMN（InitBaseBlock/GetBaseK → SetStepK4SplitMN 阶梯 → ShrinkBaseBlock 兜底 → trySplitWo → trySplitKernelAndWo）→ AdjustSmallCaseBaseBlock → DoStreamKTiling → DoLibApiTiling → PostTiling。
4. workspace：streamkType≠0 时 `2*usedCoreNum*l0c_size`（对齐段）；直调独占核下 `SetScheduleMode(1)` 语义天然满足，但 CrossCoreSetFlag 模式 0/2 的 flag 编号必须与 kernel 一致。
5. 输出清零：保留 `Conv3dDwInitOutput`（AIV Fill + flag 9），或 host 侧 aclrtMemset（仅精度验证 run；perf run 禁止）。
6. 可裁剪：GetTilingFromRepo/QueryBank 全家、HIF8、DETERMINISTIC_MODE（arch22 路径）、打印函数；首版可固定 enlarge=1（group 用例走 ORIGIN）。
7. **必须保留**：baseN 的 hkwk 对齐（§1.2）、CalculateBl1Cin1CopyLen 行数+拖尾、CalAL1Bound/CalBL1Bound（splitWo 与 tailWo 取 max）、dstStride(baseM/blockSize≤65535) 校验、initOutput + enAtomic 语义、K 分段（SINGLE_BLOCK_ADD_THRESHOLD）。

### 6.4 baseline 与对标口径
- baseline：官方 aclnn `aclnnConvolutionBackward`（outputMask 取 gradWeight；example 参考调用），同 shape 同输入
- 计时：acl event 或 msprof Task Duration；warmup 3 + perf 5 取中位数；**禁止 perf run 前 memset/memcpy device 内存**（本算子依赖输出清零语义，perf run 用 initOutput 内清零口径）
- 精度：CPU 参考实现（README 公式手写 7 层循环或 torch 反传取 weight grad）；y 恒为 fp32，atol/rtol=1e-3 量级；注意 group 场景 filter 形状 dim1 是 cin/groups

## 7. 与 ConvDx 映射的对照（迁移经验）

| 维度 | conv3d_backprop_input_v2（dx） | conv3d_backprop_filter_v2（dw） |
|---|---|---|
| M 维 | hi×wi（**输入图**） | Cout（每组，enlarge 后） |
| N 维 | Cin（每组） | **Cin×Hk×Wk**（每组；arch35 不含 Dk） |
| K 维 | HkWk×Cout1 | **Ho×Wo**（batch×Dout 在核外累加） |
| A 矩阵 | dedy（load3d 滑窗 + backpropPad） | dedy（**LoadData2D 直取**） |
| B 矩阵 | filter（rot180，load2dv2 逆序） | fmap（**load3d 滑窗 im2col**） |
| pad | backpropPad 等效变换 | 原始 pad 直喂 load3d（超限切 W/kernel） |
| 输出 | dedx 大图（Fixpipe ND/DN） | dw 小张量（**initOutput 清零 + atomic 累加**） |
| 核间切分 | splitK（Cout 轴）+ M/N 分核 | M/N/Dk/Group 分核 + **尾块 streamK**（BatchDout/HWout） |
| 对齐正确性核心 | baseK 的 HkWk 对齐 | **baseN 的 hkwk 对齐 + B1 行数** |
| 独有特性 | c04/切Hk/切HkWk/kernel split/small kernel(fuse) | groupEnlarge/splitWo/splitKernelHW/winograd/K 分段(65536)/确定性归约默认在路径上 |
| 共享 | L0/L1 基本块方法论、pingpong、load3d/load2d/Fixpipe 指令族、AIC/AIV 混合核、直调编译坑位（§6.6/6.7） | 同左 |
