# 属性枚举速查索引

本索引覆盖 AscendNPU-IR 中所有方言的枚举属性，提供枚举名、完整值列表和文档链接。

## HIVM 方言枚举

| 枚举名 | 值列表 | 文档链接 |
|--------|--------|----------|
| AddressSpace | Zero(0), GM(1), L1(2), L0A(3), L0B(4), L0C(5), UB(6) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| DataLayout | dotA_ND(1), dotB_ND(2), dotC_ND(3), nZ(4), zN(5), ND(6), Fractal(7) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| PIPE | PIPE_S(0), PIPE_V(1), PIPE_M(2), PIPE_MTE1(3), PIPE_MTE2(4), PIPE_MTE3(5), PIPE_ALL(6), PIPE_MTE4(7), PIPE_MTE5(8), PIPE_V2(9), PIPE_FIX(10), VIRTUAL_PIPE_MTE2_L1A(11), VIRTUAL_PIPE_MTE2_L1B(12), PIPE_NUM(13), PIPE_UNASSIGNED(99) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| TFuncCoreType | AIC(1), AIV(2), MIX(3), AIC_OR_AIV(4) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| TModuleCoreType | AIC(1), AIV(2), MIX(3) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| TCoreType | CUBE(1), VECTOR(2), CUBE_OR_VECTOR(3), CUBE_AND_VECTOR(4) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| PadMode | PadNull(0), PadFirstElem(1), PadValue(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| EvictionPolicy | EvictFirst(0), EvictLast(1) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| RoundMode | RINT(0), ROUND(1), FLOOR(2), CEIL(3), TRUNC(4), ODD(5), TRUNCWITHOVERFLOW(6) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| UnsignedMode | SI2SI(0), SI2UI(1), UI2SI(2), UI2UI(3) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| TypeFn | cast_signed(0), cast_unsigned(1), bitcast(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| CompareMode | EQ(0), NE(1), LT(2), GT(3), GE(4), LE(5) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| EVENT | EVENT_ID0(0), EVENT_ID1(1), EVENT_ID2(2), EVENT_ID3(3), EVENT_ID4(4), EVENT_ID5(5), EVENT_ID6(6), EVENT_ID7(7) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| UNIT_FLAG | DISABLED(0), RESERVED(1), ENABLED_WITHOUT_UPDATE(2), ENABLED_WITH_UPDATE(3) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| SyncBlockMode | ALL_CUBE(0), ALL_VECTOR(1), ALL_SUB_VECTOR(2), BARRIER_CUBE(3), BARRIER_VECTOR(4), ALL(5) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| SyncBlockInstrMode | INTER_BLOCK_SYNCHRONIZATION(0), INTER_SUBBLOCK_SYNCHRONIZATION(1), INTRA_BLOCK_SYNCHRONIZATION(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| ReduceOperation | sum(1), prod(2), max(3), min(4), max_with_index(5), min_with_index(6), any(7), all(8), xori(9), ori(10), andi(11), none(0) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| AtomicKind | NONE(0), ADD(1), MAX(2), MIN(3), AND(4), OR(5), XOR(6), CAS(7), XCHG(8) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| AlignKind | ALIGN(0), UNALIGNED(1), UNKNOWN(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| AxisKind | FIRST(0), MIDDLE(1), LAST(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| IteratorType | kParallel(0), kBroadcast(1), kTranspose(2), kReduction(3), kInterleave(4), kDeinterleave(5), kInverse(6), kPad(7), kConcat(8), kGather(9), kCumulative(10), kOpaque(99) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| DataCacheKind | ALL(0), UB(1), OUT(2), ATOMIC(3) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| DCCIMode | SINGLE_CACHE_LINE(0), ALL_CACHE_LINES(1) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| MemPlanMode | LOCAL_MEM_PLAN(0), GLOBAL_WORKSPACE_PLAN(1) | [02-memory-planning.md](../07-Memory-Management/02-memory-planning.md) |
| VFMode | SIMD(0), SIMT(1), MIX(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| FixpipePreQuantMode | NO_QUANT(0), F322F16(1), S322I8(9), QF322F32_PRE(15), F322BF16(16) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| FixpipePreReluMode | NO_RELU(0), NORMAL_RELU(1), LEAKY_RELU(2), P_RELU(3) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| FixpipeDualDstMode | NO_DUAL(0), ROW_SPLIT(1), COLUMN_SPLIT(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| FixpipeDMAMode | NZ2ND(0), NZ2DN(1), NZ2NZ(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| DeinterleaveMode | CHANNEL_0(0), CHANNEL_1(1), ALL_CHANNELS(999) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| DescaleMode | DescaleNull(0), DescalePerChannel(1), DescalePerTensor(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| MatmulBiasMode | NoBias(0), PerChannelAdd(1), PostPerChannelAddWithSplitK(2), ElementwiseAdd(3), MMInitPerChannelAddWithSplitK(4) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| PaddingOption | PAD_ZERO(1), PAD_NAN(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| CacheModifier | NONE(1), CA(2), CG(3), WB(4), CS(5), WT(6), CV(7) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |
| MemoryEffect | READ(0), WRITE(1), READ_WRITE(2) | [01-enumerations.md](../01-HIVM-Dialect/06-Attributes-Types/01-enumerations.md) |

## HACC 方言枚举

| 枚举名 | 值列表 | 文档链接 |
|--------|--------|----------|
| HACCFuncType | HOST(1), DEVICE(2) | [02-device-specification.md](../02-HACC-Dialect/02-device-specification.md) |
| KernelArgType | kFFTSBaseAddr(0), kInput(1), kOutput(2), kInputAndOutput(3), kWorkspace(4), kSyncBlockLock(5), kTilingKey(6), kTilingData(7), kTilingStruct(8), kMeshArg(9), kSanitizerAddr(10), kGMAddr(11) | [03-kernel-args.md](../02-HACC-Dialect/03-kernel-args.md) |
| HostFuncType | kEntry(1), kTilingFunction(2), kInferOutputShapeFunction(3), kInferWorkspaceShapeFunction(4), kInferSyncBlockLockNumFunction(5), kInferSyncBlockLockInitFunction(6), kInferVFModeFunction(7), kGetTilingStructSizeFunction(8) | [02-device-specification.md](../02-HACC-Dialect/02-device-specification.md) |
| DeviceSpec | AI_CORE_COUNT(0), CUBE_CORE_COUNT(1), VECTOR_CORE_COUNT(2), UB_SIZE(3), L1_SIZE(4), L0A_SIZE(5), L0B_SIZE(6), L0C_SIZE(7), UB_ALIGN_SIZE(8), L1_ALIGN_SIZE(9), L0C_ALIGN_SIZE(10), MINIMAL_D_CACHE_SIZE(11), MAXIMUM_D_CACHE_SIZE(12), ARCH(13) | [03-hardware-specs-ir.md](03-hardware-specs-ir.md) |

## HFusion 方言枚举

| 枚举名 | 值列表 | 文档链接 |
|--------|--------|----------|
| FusionKind | (由 HFusion_FusionKindEnum 定义) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| UnaryFn | relu(0), sqrt(1), rsqrt(2), rec(3), vnot(4), tanh(5), sin(6), cos(7), atan(8), tan(9), absi(10), erf(11), log2(12), log10(13), log1p(14), exp2(15), expm1(16), ilogb(17) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| BinaryFn | vor(0), vand(1), vxor(2), minf(3), maxf(4), powf(5), mod(6), shli(7), shrsi(8), shrui(9), ldexp(10), ceildivsi(11), ceildivui(12), floordivsi(13), powi(14), minnumf(15), maxnumf(16), modui(17), divfhp(18) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| CompareFn | veq(0), vne(1), vle(2), vlt(3), vge(4), vgt(5), vule(6), vuge(7), vugt(8), vult(9) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| TernaryFn | select(0) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| TypeFn | cast_signed(0), cast_unsigned(1), bitcast(2) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| RoundMode | RINT(0), ROUND(1), FLOOR(2), CEIL(3), TRUNC(4), ODD(5), TRUNCWITHOVERFLOW(6) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| UnsignedMode | SI2SI(0), SI2UI(1), UI2SI(2), UI2UI(3) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| ReduceWithIndexKind | (由 ReduceWithIndexKind 定义) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| PaddingOption | PAD_ZERO(1), PAD_NAN(2) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| CacheModifier | NONE(1), CA(2), CG(3), WB(4), CS(5), WT(6), CV(7) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |
| EvictionPolicy | NORMAL(1), EVICT_FIRST(2), EVICT_LAST(3) | [07-attributes-enums.md](../03-HFusion-Dialect/07-attributes-enums.md) |

## 源码参考

- HIVM 枚举定义：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- HACC 枚举定义：[HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td)
- HFusion 枚举定义：[HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td), [HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td)
