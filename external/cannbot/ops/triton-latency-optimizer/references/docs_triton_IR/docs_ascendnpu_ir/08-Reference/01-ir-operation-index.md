# IR 操作速查索引

本索引覆盖 AscendNPU-IR 中所有方言的所有操作，提供名称到文档的快速链接。

## HIVM 方言

### DMA 操作

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.load | HIVM | [01-load.md](../01-HIVM-Dialect/01-DMA-Operations/01-load.md) |
| hivm.hir.store | HIVM | [02-store.md](../01-HIVM-Dialect/01-DMA-Operations/02-store.md) |
| hivm.hir.copy | HIVM | [05-copy.md](../01-HIVM-Dialect/01-DMA-Operations/05-copy.md) |
| hivm.hir.fixpipe | HIVM | [06-fixpipe.md](../01-HIVM-Dialect/01-DMA-Operations/06-fixpipe.md) |
| hivm.hir.nd2nz | HIVM | [03-nd2nz.md](../01-HIVM-Dialect/01-DMA-Operations/03-nd2nz.md) |
| hivm.hir.nz2nd | HIVM | [04-nz2nd.md](../01-HIVM-Dialect/01-DMA-Operations/04-nz2nd.md) |
| hivm.hir.l12ub | HIVM | [01-load.md](../01-HIVM-Dialect/01-DMA-Operations/01-load.md) |
| hivm.hir.atomic_cas | HIVM | [07-atomic.md](../01-HIVM-Dialect/01-DMA-Operations/07-atomic.md) |
| hivm.hir.atomic_xchg | HIVM | [07-atomic.md](../01-HIVM-Dialect/01-DMA-Operations/07-atomic.md) |

### 间接访问操作

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.gather_load | HIVM | [08-gather-scatter.md](../01-HIVM-Dialect/01-DMA-Operations/08-gather-scatter.md) |
| hivm.hir.scatter_store | HIVM | [08-gather-scatter.md](../01-HIVM-Dialect/01-DMA-Operations/08-gather-scatter.md) |
| hivm.hir.local_load | HIVM | [08-gather-scatter.md](../01-HIVM-Dialect/01-DMA-Operations/08-gather-scatter.md) |
| hivm.hir.local_store | HIVM | [08-gather-scatter.md](../01-HIVM-Dialect/01-DMA-Operations/08-gather-scatter.md) |
| hivm.hir.indirect_load | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |
| hivm.hir.indirect_store | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |
| hivm.hir.gatherT | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |
| hivm.hir.scatterT | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |
| hivm.hir.index_put | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |
| hivm.hir.embedding_gather | HIVM | [09-indirect-access.md](../01-HIVM-Dialect/01-DMA-Operations/09-indirect-access.md) |

### 向量操作 — 一元

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.vexp | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vabs | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vln | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vrelu | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vrsqrt | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vsqrt | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vtanh | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vsin | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vcos | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.verf | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vrec | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |
| hivm.hir.vnot | HIVM | [01-unary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/01-unary-ops.md) |

### 向量操作 — 二元

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.vadd | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmul | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmulext | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vdiv | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmax | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmin | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vor | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vand | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vxor | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmod | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vmodui | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |
| hivm.hir.vpow | HIVM | [02-binary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/02-binary-ops.md) |

### 向量操作 — 移位

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.vshl | HIVM | [06-shift-ops.md](../01-HIVM-Dialect/02-Vector-Operations/06-shift-ops.md) |
| hivm.hir.vshr | HIVM | [06-shift-ops.md](../01-HIVM-Dialect/02-Vector-Operations/06-shift-ops.md) |

### 向量操作 — 比较/三元/类型转换

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.vcmp | HIVM | [05-compare-ops.md](../01-HIVM-Dialect/02-Vector-Operations/05-compare-ops.md) |
| hivm.hir.vsel | HIVM | [03-ternary-ops.md](../01-HIVM-Dialect/02-Vector-Operations/03-ternary-ops.md) |
| hivm.hir.vcast | HIVM | [04-cast-ops.md](../01-HIVM-Dialect/02-Vector-Operations/04-cast-ops.md) |
| hivm.hir.vbrc | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |

### 向量操作 — 归约/数据移动/累积排序

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.vreduce | HIVM | [07-reduction-ops.md](../01-HIVM-Dialect/02-Vector-Operations/07-reduction-ops.md) |
| hivm.hir.vtranspose | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vinterleave | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vdeinterleave | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vflip | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vmulextended | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vpad | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vconcat | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vgather | HIVM | [08-data-movement.md](../01-HIVM-Dialect/02-Vector-Operations/08-data-movement.md) |
| hivm.hir.vcumprod | HIVM | [09-cumulative-sort.md](../01-HIVM-Dialect/02-Vector-Operations/09-cumulative-sort.md) |
| hivm.hir.vcumsum | HIVM | [09-cumulative-sort.md](../01-HIVM-Dialect/02-Vector-Operations/09-cumulative-sort.md) |
| hivm.hir.vsort | HIVM | [09-cumulative-sort.md](../01-HIVM-Dialect/02-Vector-Operations/09-cumulative-sort.md) |

### Macro 操作

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.mmadL1 | HIVM | [01-mmad-l1.md](../01-HIVM-Dialect/03-Macro-Operations/01-mmad-l1.md) |
| hivm.hir.batchMmadL1 | HIVM | [02-batch-mmad-l1.md](../01-HIVM-Dialect/03-Macro-Operations/02-batch-mmad-l1.md) |
| hivm.hir.matmul | HIVM | [03-matmul.md](../01-HIVM-Dialect/03-Macro-Operations/03-matmul.md) |
| hivm.hir.mix_matmul | HIVM | [04-mix-matmul.md](../01-HIVM-Dialect/03-Macro-Operations/04-mix-matmul.md) |
| hivm.hir.mix_group_matmul | HIVM | [05-mix-group-matmul.md](../01-HIVM-Dialect/03-Macro-Operations/05-mix-group-matmul.md) |

### 同步操作

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.set_flag | HIVM | [03-unit-flag.md](../01-HIVM-Dialect/04-Synchronization/03-unit-flag.md) |
| hivm.hir.wait_flag | HIVM | [03-unit-flag.md](../01-HIVM-Dialect/04-Synchronization/03-unit-flag.md) |
| hivm.hir.pipe_barrier | HIVM | [01-pipe-sync.md](../01-HIVM-Dialect/04-Synchronization/01-pipe-sync.md) |
| hivm.hir.sync_block | HIVM | [02-block-sync.md](../01-HIVM-Dialect/04-Synchronization/02-block-sync.md) |
| hivm.hir.sync_block_set | HIVM | [02-block-sync.md](../01-HIVM-Dialect/04-Synchronization/02-block-sync.md) |
| hivm.hir.sync_block_wait | HIVM | [02-block-sync.md](../01-HIVM-Dialect/04-Synchronization/02-block-sync.md) |

### 其他 HIVM 操作

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hivm.hir.get_block_idx | HIVM | [00-overview.md](../01-HIVM-Dialect/00-overview.md) |
| hivm.hir.get_block_num | HIVM | [00-overview.md](../01-HIVM-Dialect/00-overview.md) |
| hivm.hir.get_sub_block_idx | HIVM | [00-overview.md](../01-HIVM-Dialect/00-overview.md) |
| hivm.hir.get_sub_block_num | HIVM | [00-overview.md](../01-HIVM-Dialect/00-overview.md) |
| hivm.hir.set_atomic | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.set_mask_norm | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.set_ctrl | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.load_scalar | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.dcci | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.convert_layout | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.pointer_cast | HIVM | [00-overview.md](../07-Memory-Management/02-memory-planning.md) |
| hivm.hir.bitcast | HIVM | [04-cast-ops.md](../01-HIVM-Dialect/02-Vector-Operations/04-cast-ops.md) |
| hivm.hir.set_ffts_base_addr | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.debug | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.init_debug | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.finish_debug | HIVM | [10-special-ops.md](../01-HIVM-Dialect/02-Vector-Operations/10-special-ops.md) |
| hivm.hir.custom | HIVM | [01-custom-op.md](../01-HIVM-Dialect/05-Custom-Operations/01-custom-op.md) |
| hivm.hir.custom_macro | HIVM | [02-custom-macro-op.md](../01-HIVM-Dialect/05-Custom-Operations/02-custom-macro-op.md) |

## HFusion 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hfusion.print | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.assert | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.barrier | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.symbolic_dim | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.mulext | HFusion | [01-elementwise-ops.md](../03-HFusion-Dialect/01-elementwise-ops.md) |
| hfusion.interleave | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.deinterleave | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.flip | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.isinf | HFusion | [01-elementwise-ops.md](../03-HFusion-Dialect/01-elementwise-ops.md) |
| hfusion.isnan | HFusion | [01-elementwise-ops.md](../03-HFusion-Dialect/01-elementwise-ops.md) |
| hfusion.isfinite | HFusion | [01-elementwise-ops.md](../03-HFusion-Dialect/01-elementwise-ops.md) |
| hfusion.cumsum | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.cumprod | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.atomic_cas | HFusion | [05-memory-ops.md](../03-HFusion-Dialect/05-memory-ops.md) |
| hfusion.atomic_xchg | HFusion | [05-memory-ops.md](../03-HFusion-Dialect/05-memory-ops.md) |
| hfusion.sort | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.embedding_gather | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.gather_load | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.scatter_store | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.indirect_load | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.indirect_store | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.gatherT | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.index_put | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.scatterT | HFusion | [04-data-movement-ops.md](../03-HFusion-Dialect/04-data-movement-ops.md) |
| hfusion.histogram | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |
| hfusion.matmul_mx | HFusion | [03-matmul-ops.md](../03-HFusion-Dialect/03-matmul-ops.md) |
| hfusion.reduce_with_index | HFusion | [02-reduction-ops.md](../03-HFusion-Dialect/02-reduction-ops.md) |
| hfusion.arange | HFusion | [06-special-ops.md](../03-HFusion-Dialect/06-special-ops.md) |

## HACC 方言

HACC 方言主要定义属性和接口，不定义独立操作。其核心功能通过属性标注 `func.func` 和 `ModuleOp` 实现。

详见 [02-HACC-Dialect](../02-HACC-Dialect/00-overview.md)

## Scope 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| scope.scope | Scope | [01-scope-dialect.md](../04-Other-Dialects/01-scope-dialect.md) |
| scope.return | Scope | [01-scope-dialect.md](../04-Other-Dialects/01-scope-dialect.md) |

## Symbol 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| symbol.symbolic_int | Symbol | [02-symbol-dialect.md](../04-Other-Dialects/02-symbol-dialect.md) |
| symbol.bind_symbolic_shape | Symbol | [02-symbol-dialect.md](../04-Other-Dialects/02-symbol-dialect.md) |

## MemRefExt 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| memref_ext.alloc_workspace | MemRefExt | [03-memrefext-dialect.md](../04-Other-Dialects/03-memrefext-dialect.md) |

## MathExt 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| math_ext.ilogb | MathExt | [04-mathext-dialect.md](../04-Other-Dialects/04-mathext-dialect.md) |
| math_ext.ldep | MathExt | [04-mathext-dialect.md](../04-Other-Dialects/04-mathext-dialect.md) |
| math_ext.divfhp | MathExt | [04-mathext-dialect.md](../04-Other-Dialects/04-mathext-dialect.md) |

## HMAP 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| hmap.all_to_all_v | HMAP | [05-hmap-dialect.md](../04-Other-Dialects/05-hmap-dialect.md) |

## Annotation 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| annotation.mark | Annotation | [06-annotation-dialect.md](../04-Other-Dialects/06-annotation-dialect.md) |

## AscendDPX 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| ascend_dpx.thread_id_x | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.thread_id_y | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.thread_id_z | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_idx_x | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_idx_y | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_idx_z | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_idx | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_dim_x | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_dim_y | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.block_dim_z | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.grid_dim_x | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.grid_dim_y | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.grid_dim_z | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.clock32 | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.clock64 | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.core_id | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.load | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.store | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.sync_threads | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.shfl.up | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.shfl.down | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.shfl.bfly | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.shfl.idx | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_and | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_or | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_xor | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_inc | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_dec | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_max | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_min | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_add | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_sub | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_umax | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_umin | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_exchange | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atomic_cas | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.cast | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.atan | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.ceil | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.cos | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.erf | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.exp2 | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.exp | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.floor | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.ilogb | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.log1p | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.log2 | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.log | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.recip | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.relu | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.rint | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.round | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.rsqrt | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.sin | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.sqrt | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.tanh | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.tan | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.div | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.udiv | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.pow | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.umulhi | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.ldexp | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.isfinite | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.isinf | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |
| ascend_dpx.isnan | AscendDPX | [07-ascend-dpx-dialect.md](../04-Other-Dialects/07-ascend-dpx-dialect.md) |

## Triton 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| tt.int_to_ptr | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.ptr_to_int | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.bitcast | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.fp_to_fp | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.clampf | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.precise_sqrt | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.precise_divf | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.mulhiui | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.addptr | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.advance | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.load | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.store | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.atomic_rmw | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.atomic_cas | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.splat | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.unsplat | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.expand_dims | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.reshape | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.broadcast | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.cat | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.join | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.split | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.trans | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.get_program_id | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.get_num_programs | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.dot | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.dot_scaled | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.extern_elementwise | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.make_range | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.elementwise_inline_asm | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.histogram | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.gather | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.print | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.assert | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.make_tensor_ptr | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.make_tensor_descriptor | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.call | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.func | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.return | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.descriptor_load | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.descriptor_store | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.descriptor_reduce | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.descriptor_gather | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |
| tt.descriptor_scatter | Triton | [01-triton-ops.md](../05-Triton-Dialects/01-triton-ops.md) |

## TritonGPU 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| tritongpu.convert_layout | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.async_wait | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.async_commit_group | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.async_copy_global_to_local | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.local_alloc | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.local_dealloc | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.memdesc_index | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.memdesc_subslice | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.memdesc_trans | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.memdesc_reshape | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.memdesc_reinterpret | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.local_load | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.local_store | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.fp4_to_fp | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.global_scratch_alloc | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.warp_specialize | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.warp_specialize.partitions | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.warp_yield | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |
| tritongpu.warp_return | TritonGPU | [03-tritongpu-ops.md](../05-Triton-Dialects/03-tritongpu-ops.md) |

## Gluon 方言

| 操作名 | 方言 | 文档链接 |
|--------|------|----------|
| gluon.set_auto_layout | Gluon | [05-gluon-dialect.md](../05-Triton-Dialects/05-gluon-dialect.md) |

## 源码参考

- HIVM 操作定义：[HIVM IR 目录](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/)
- HFusion 操作定义：[HFusion IR 目录](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/)
- Triton 操作定义：[TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonOps.td)
- TritonGPU 操作定义：[TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)
- Gluon 操作定义：[GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Gluon/IR/GluonOps.td)
- AscendDPX 操作定义：[AscendDPXOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXOps.td)
