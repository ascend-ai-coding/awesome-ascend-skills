# Transpose 性能调优参考(策略文档 + 可复用 kernel 模板)

本目录为 Transpose 算子(AscendC / arch35 / dav-3510)提供**按 tiling 策略组织的调优参考**:开发者或 agent 根据实际入参选定策略后,可直接引用对应 kernel 模板,或仅做微调即可完成调优。每份策略文档均包含两部分:**一、适用场景**(为何需要、解决什么问题、为何提速)与 **二、kernel 执行流程与关键实现**(关键函数流程 + 所用关键技术)。

## 目录结构

```
conversion/
├── guide.md                      # 本文:调优工作流入口 + 策略总览 + 关键技术对比
├── transpose_tiling_data.md      # TilingData 结构与各参数说明
├── transpose_<策略>.md           # 每种策略一份原理与实现文档
├── transpose_fusion_design.md    # 融合设计文档
└── templates/                    # 可复用 kernel 模板代码
    ├── transpose_tiling_data.template   # 所有 TilingData POD 结构定义
    └── dav3510/                        # 各策略 kernel 模板实现(.template 文件)
        ├── transpose_base.template
        ├── transpose_tensor_move.template
        ├── transpose_small_shape.template
        ├── transpose_cut_one_axis.template
        ├── transpose_cut_two_axis.template
        ├── transpose_n_last.template
        ├── transpose_big_dim.template
        ├── transpose_with_gather.template
        ├── transpose_transdata_5hd.template
        └── transpose_transdata_5hd_021axis.template
```

## 调优工作流

1. **判定入参**:确定 dtype(元素字节宽度)、输入 shape、perm(转置轴序)。
2. **选策略**:对照下方「策略速查」表,按命中条件确定 tiling 策略。
3. **读原理**:阅读该策略的 `transpose_<策略>.md`,理解适用场景与 kernel 执行流程。
4. **引用代码**:从 [`templates/dav3510/`](templates/dav3510/) 取对应策略的 `.template` 模板文件,按实际算子微调后接入。
5. **填 TilingData**:参照 [`transpose_tiling_data.md`](transpose_tiling_data.md) 填充对应结构;结构体定义见 [`templates/transpose_tiling_data.template`](templates/transpose_tiling_data.template)。
6. **SIMT 策略专项检查**(仅 SMALL_SHAPE 策略):若使用 SMALL_SHAPE 策略,务必检查 memory coalescing 和向量化读写。原始模板逐元素读写带宽利用率仅 6.25%,当 perm 末轴不转置时(`perm[permSize-1] == permSize-1`)应实现向量化读写(每线程处理 C 个连续元素)。详见 [`transpose_small_shape.md` §6](transpose_small_shape.md#6-memory-coalescing-与向量化读写)。
7. **N_LAST 策略维度合并检查**(仅 N_LAST 策略):若使用 N_LAST 策略,检查 CopyOut 的循环顺序是否与 output 连续性匹配。当 perm 导致相邻 input 轴在 output 中不连续时,CopyOut 退化为多次 strided 小搬运。可通过检查 `inv_perm` 连续性合并 output 中相邻维度组,将 strided 多循环写出降为连续单次写出。详见 [`transpose_n_last.md` §3](transpose_n_last.md#三copyout-维度合并优化)。

## 策略速查

各策略对应的 kernel 模板文件、参考文档与 TilingData 结构如下:

| 策略 | 命中条件(简) | kernel 模板文件 | 参考文档 | TilingData 结构 |
|------|----------------|-----------------|----------|-----------------|
| TENSOR_MOVE | 规约后 `dim==1`,退化为纯拷贝 | `templates/dav3510/transpose_tensor_move.template` | [transpose_tensor_move.md](transpose_tensor_move.md) | `TransposeOpTilingData` |
| SMALL_SHAPE | 总字节数 < 阈值(小 shape),SIMT 逐元素 | `templates/dav3510/transpose_small_shape.template` | [transpose_small_shape.md](transpose_small_shape.md) | `TransposeOpTilingData` |
| CUT_ONCE | 单轴切分即可满足输入/输出 UB 约束 | `templates/dav3510/transpose_cut_one_axis.template` | [transpose_cut_one_axis.md](transpose_cut_one_axis.md) | `TransposeOpTilingData` |
| CUT_TWICE | 需在输入、输出两方向各切一轴 | `templates/dav3510/transpose_cut_two_axis.template` | [transpose_cut_two_axis.md](transpose_cut_two_axis.md) | `TransposeOpTilingData` |
| N_LAST_TRANSPOSE | 末轴不参与转置且末轴元素数 ≥32 | `templates/dav3510/transpose_n_last.template` | [transpose_n_last.md](transpose_n_last.md) | `TransposeOpTilingData` |
| BIG_DIM | 规约后维度 > NDDMA 上限(5) | `templates/dav3510/transpose_big_dim.template` | [transpose_big_dim.md](transpose_big_dim.md) | `TransposeOpTilingData` |
| GATHER_TRANSPOSE | 末轴参与转置,UB 内 gather 重排(需 dav-3510 + `TRANSPOSE_ENABLE_GATHER`) | `templates/dav3510/transpose_with_gather.template` | [transpose_gather.md](transpose_gather.md) | `GatherTransposeTilingData` |
| VCONV_TRANSPOSE (5HD) | 2D 末轴交换、16bit、R>5 | `templates/dav3510/transpose_transdata_5hd.template` | [transpose_vconv_5hd.md](transpose_vconv_5hd.md) | `TransposeVCONVTilingData` |
| VCONV_021_TRANSPOSE | perm=[0,2,1] 保 batch 转置,支持 8/16/32bit | `templates/dav3510/transpose_transdata_5hd_021axis.template` | [transpose_vconv_021.md](transpose_vconv_021.md) | `Transpose021VCONVTilingData` |

> NDDMA 家族(TENSOR_MOVE / SMALL_SHAPE / CUT_ONCE / CUT_TWICE / N_LAST / BIG_DIM)共用同一个 `TransposeOpTilingData`;加速策略(GATHER / VCONV_5HD / VCONV_021)各有独立结构。所有结构均 `#pragma pack(push, 8)` 8 字节对齐,定义见 [`templates/transpose_tiling_data.template`](templates/transpose_tiling_data.template)。

## 关键技术一览(便于横向对比)

- **NDDMA 多维 DMA**:CUT_ONCE / CUT_TWICE / BIG_DIM / N_LAST —— 用带 stride 的多维描述符在搬运中完成 perm 取址。
- **VCONV / TransDataTo5HD 向量转置**:VCONV_5HD / VCONV_021 —— 片上 16×16 块转置。
- **MicroAPI / regbase gather**:GATHER —— `DataCopyGatherImpl` 按预生成 index 在 UB 内重排(仅 dav-3510)。
- **SIMT 逐元素**:SMALL_SHAPE —— `asc_vf_call` + 魔数除法反算索引,省搬运流水固定开销。**注意**:原始模板逐元素读写带宽利用率仅 6.25%,当 perm 末轴不转置时应实现向量化读写(每线程处理 C 个连续元素),详见 [transpose_small_shape.md §6](transpose_small_shape.md#6-memory-coalescing-与向量化读写)。
- **double buffer**:TENSOR_MOVE / N_LAST / VCONV 系列 / GATHER 用到;CUT_ONCE / CUT_TWICE / BIG_DIM 为单缓冲(见各文档说明)。
- **CopyOut 维度合并**:N_LAST —— 当 perm 导致循环顺序与 output 连续性不匹配时,可通过检查 `inv_perm` 连续性合并 output 中相邻的维度组,将 strided 多循环写出降为连续单次写出。详见 [transpose_n_last.md §3](transpose_n_last.md#三copyout-维度合并优化)。

## 编译注意

- GATHER 策略受编译宏 `TRANSPOSE_ENABLE_GATHER`(默认 0)保护,需 MicroAPI/Reg 寄存器模型,仅 `__NPU_ARCH__==3510` 可用。关闭时 host 不产出该策略,相应 shape 由 NDDMA 家族覆盖。
- kernel 模板依赖 CANN toolkit 头(`kernel_operator.h`、`op_kernel/platform_util.h`、`op_kernel/math_util.h`、`simt_api/asc_simt.h`、`reg_compute/*`),详见 [`transpose_tiling_data.md`](transpose_tiling_data.md#外部依赖cann-toolkit-头文件)。
- VCONV 5HD 策略仅支持 16bit;021 VCONV 策略支持 8/16/32bit。
