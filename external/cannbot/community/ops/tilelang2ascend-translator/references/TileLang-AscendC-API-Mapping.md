# TileLang-Ascend API Mapping

本文整理 TileLang Ascend 编程中常见 API 与最终落到的 AscendC API 的对应关系。

说明：
- 对于部分 API，tensor-tensor 和 tensor-scalar 的 lowering 不同，分别列出。

## 数据搬运

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `T.copy(src, dst)` (`GM -> L1`) | `AscendC::DataCopy(...)` | 常见为 `DataCopy(..., Nd2NzParams)`；若目标是 `TQue`，配合 `AllocTensor` 和 `EnQue`。 |
| `T.copy(src, dst)` (`L1 -> L0A`) | `AscendC::LoadData(...)` / `AscendC::LoadDataWithTranspose(...)` | 是否转置取决于 layout。 |
| `T.copy(src, dst)` (`L1 -> L0B`) | `AscendC::LoadData(...)` / `AscendC::LoadDataWithTranspose(...)` | 是否转置取决于 layout。 |
| `T.copy(src, dst)` (`L0C -> GM`) | `AscendC::Fixpipe(...)` |  |
| `T.copy(src, dst)` (`GM -> UB`) | `AscendC::DataCopyPad(local, global, copyParams, padParams)` | 若目标是 `TQue`，配合 `AllocTensor` 和 `EnQue`；如果是矩阵的二维子 tile，常改用 `DataCopy` + `DataCopyParams`。 |
| `T.copy(src, dst)` (`UB -> GM`) | `AscendC::DataCopyPad(global, local, copyParams)` | 若来源是 `TQue`，配合 `DeQue` 和 `FreeTensor`；如果是写回矩阵的二维子 tile，常改用 `DataCopy` + `DataCopyParams`。 |
| `T.copy(src, dst)` (`UB -> UB`, same dtype) | `AscendC::DataCopy(...)` |  |
| `T.copy(src, dst)` (`UB -> UB`, cast dtype) | `AscendC::Cast(...)` |  |

### DataCopy / DataCopyPad 注意事项

- `DataCopyPad` 的 `GM -> UB` 和 `UB -> GM` 参数个数不同：
  - `GM -> UB`：`DataCopyPad(local, global, copyParams, padParams)`，需要 4 参数，并传入 `DataCopyPadExtParams<T>`。
  - `UB -> GM`：`DataCopyPad(global, local, copyParams)`，只需要 3 参数，不传 `padParams`。
- `DataCopyExtParams` 常用字段为 `blockCount, blockLen, srcStride, dstStride, rsv`；其中 `blockLen` 单位是字节，`srcStride`/`dstStride` 的单位随 GM 或 Local 位置变化。
- `DataCopyPadExtParams<T>` 常用字段为 `isPad, leftPadding, rightPadding, padValue`；`isPad` 属于 `DataCopyPadExtParams<T>`，不要写进 `DataCopyExtParams`。
- 参与 `GM <-> UB` 搬运的 UB buffer 优先映射为 `TQue` 分配的 `LocalTensor`，并按方向配合 `AllocTensor`/`EnQue` 或 `DeQue`/`FreeTensor`；不要把直接参与 `DataCopy` / `DataCopyPad` 的 buffer 映射为普通 `TBuf`。
- 连续且 32B 对齐的搬运优先用 `DataCopy`；非对齐、尾块或需要补齐时再用 `DataCopyPad`。
- 矩阵二维子 tile 或带 stride 的搬运应显式构造 `DataCopyParams` / `DataCopyExtParams`。
- 普通 `UB -> UB` 搬运优先用 `DataCopy`；`Local -> Local` 的 `DataCopyPad + Nd2NzParams` 属于特殊格式转换路径，不按普通 `T.copy(UB -> UB)` 套用。

## 向量与逐元素计算

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `T.tile.add(dst, src0, src1)` | `AscendC::Add(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.add(dst, src0, scalar)` | `AscendC::Adds(...)` | `src1` 为 scalar 或 `BufferLoad`。 |
| `T.tile.sub(dst, src0, src1)` | `AscendC::Sub(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.sub(dst, src0, scalar)` | `AscendC::Adds(...)` | 通过对标量取负实现。 |
| `T.tile.mul(dst, src0, src1)` | `AscendC::Mul(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.mul(dst, src0, scalar)` | `AscendC::Muls(...)` | `src1` 为 scalar 或 `BufferLoad`。 |
| `T.tile.div(dst, src0, src1)` | `AscendC::Div(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.div(dst, src0, scalar)` | `AscendC::Muls(...)` | 通过乘以倒数实现。 |
| `T.tile.max(dst, src0, src1)` | `AscendC::Max(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.max(dst, src0, scalar)` | `AscendC::Maxs(...)` | `src1` 为 scalar 或 `BufferLoad`。 |
| `T.tile.min(dst, src0, src1)` | `AscendC::Min(...)` | `src1` 为 tensor / buffer。 |
| `T.tile.min(dst, src0, scalar)` | `AscendC::Mins(...)` | `src1` 为 scalar 或 `BufferLoad`。 |
| `T.tile.exp(dst, src0)` | `AscendC::Exp(...)` |  |
| `T.tile.sigmoid(dst, src, tmp)` | `AscendC::Sigmoid(...)` |  |
| `T.tile.ln(dst, src0)` | `AscendC::Ln(...)` |  |
| `T.tile.abs(dst, src0)` | `AscendC::Abs(...)` |  |
| `T.tile.pow(dst, src0, src1, tmp)` | `AscendC::Power(...)` |  |
| `T.tile.reciprocal(dst, src0)` | `AscendC::Reciprocal(...)` |  |
| `T.tile.sqrt(dst, src0)` | `AscendC::Sqrt(...)` |  |
| `T.tile.rsqrt(dst, src0)` | `AscendC::Rsqrt(...)` |  |
| `T.tile.relu(dst, src0)` | `AscendC::Relu(...)` |  |
| `T.tile.leaky_relu(dst, src0, scalar)` | `AscendC::LeakyRelu(...)` |  |
| `T.tile.axpy(dst, src0, scalar)` | `AscendC::Axpy(...)` |  |
| `T.tile.sin(dst, src0, tmp)` | `AscendC::Sin(...)` |  |
| `T.tile.cos(dst, src0, tmp)` | `AscendC::Cos(...)` |  |
| `T.tile.bitwise_and(dst, src0, src1)` | `AscendC::And(...)` |  |
| `T.tile.bitwise_or(dst, src0, src1)` | `AscendC::Or(...)` |  |
| `T.tile.bitwise_not(dst, src0)` | `AscendC::Not(...)` |  |
| `T.tile.bitwise_xor(dst, src0, src1)` | `AscendC::Xor(...)` |  |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | `AscendC::ShiftLeft(...)` |  |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | `AscendC::ShiftRight(...)` |  |
| `T.tile.compare(dst, src0, src1, mode)` | `AscendC::Compare(...)` | tensor-tensor compare。 |
| `T.tile.compare(dst, src0, scalar, mode)` | `AscendC::CompareScalar(...)` | tensor-scalar compare。 |
| `T.tile.select(dst, selMask, src0, src1, selMode)` | `AscendC::Select(...)` |  |
| `T.tile.cast(dst, src, mode, count)` | `AscendC::Cast(...)` |  |
| `T.tile.round(dst, src, tmp, count)` | `AscendC::Round(...)` |  |
| `T.tile.transpose(dst, src)` | `AscendC::Transpose(...)` |  |
| `T.tile.broadcast(dst, src, tmp)` | `AscendC::Broadcast<...>(...)` | broadcast axis 由 shape 自动推断。 |
| `T.tile.createvecindex(dst, first_value)` | `AscendC::CreateVecIndex(...)` |  |
| `T.tile.fill(dst, value)` | `AscendC::Duplicate(...)` |  |
| `T.tile.arith_progression(dst, first_value, diff_value, count)` | `AscendC::ArithProgression(...)` |  |
| `T.tile.clamp_max(dst, src, tmp, scalar, count)` | `AscendC::ClampMax(...)` |  |
| `T.tile.clamp_min(dst, src, tmp, scalar, count)` | `AscendC::ClampMin(...)` |  |
| `T.tile.clamp(dst, src, tmp, min_scalar, max_scalar, count)` | `AscendC::ClampMin(...)` + `AscendC::ClampMax(...)` | 先 clamp min，再 clamp max。 |
| `T.tile.bilinear_interpolation(dst, src0, src0_offset, src1, mask, h_repeat, repeat_mode, dst_blk_stride, v_r_offset, v_repeat, shared_tmp_buffer)` | `AscendC::BilinearInterpolation(...)` |  |
| `T.tile.brcb(dst, src0, repeat_time, dst_blk_stride, dst_rep_stride)` | `AscendC::Brcb(...)` |  |

## 排序、Gather 与索引相关

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `T.tile.sort(dst, src, indices, tmp_buffer, repeat_time)` | `AscendC::Sort(...)` |  |
| `T.tile.sort32(dst, src0, src1)` | `AscendC::Sort32(...)` |  |
| `T.tile.init_sort_buf(src, ele_num, rsv=0)` | `AscendC::Duplicate(...)` | 用若干次 `Duplicate` 初始化排序缓冲区。 |
| `T.tile.merge_sort(dst, src, block_size, block_num, is_copy)` | `AscendC::MrgSort(...)` |  |
| `T.tile.merge_sort(dst, src, block_size, block_num, 1)` | `AscendC::MrgSort(...)` + `AscendC::DataCopy(...)` | `is_copy = 1` 时会追加 copy。 |
| `T.tile.topk(dst, src, tmp_buffer, block_size)` | `AscendC::MrgSort(...)` + `AscendC::DataCopy(...)` |  |
| `T.tile.gather(dst, src, src_offset, src_base_addr)` | `AscendC::Gather(...)` |  |
| `T.tile.gatherb(dst, src0, offset, repeat_time, dst_blk_stride, dst_rep_stride)` | `AscendC::Gatherb(...)` |  |
| `T.tile.gather_mask(dst, src, "P0101"/...)` | `GatherMask(...)` | 固定 pattern 模式，走封装实现，不是直接打印成 `AscendC::GatherMask` 字符串。 |
| `T.tile.gather_mask(dst, src, pattern_buf)` | `AscendC::Muls(...)` + `AscendC::Gather(...)` | 自定义 pattern 会先把索引乘元素字节数，再做 gather。 |

## Reduce

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `T.reduce.sum(...)` / Tile reduce sum | `AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(...)` / `AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(...)` | 按 `dim` 选择 AR/RA。 |
| `T.reduce.max(...)` / Tile reduce max | `AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(...)` / `AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(...)` | 按 `dim` 选择 AR/RA。 |
| `T.reduce.min(...)` / Tile reduce min | `AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(...)` / `AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(...)` | 按 `dim` 选择 AR/RA。 |
| `T.tile.block_reduce_sum(dst, src, ...)` | `AscendC::BlockReduceSum(...)` |  |
| `T.tile.block_reduce_max(dst, src, ...)` | `AscendC::BlockReduceMax(...)` |  |
| `T.tile.block_reduce_min(dst, src, ...)` | `AscendC::BlockReduceMin(...)` |  |
| `T.tile.wholereducesum(dst, src, ...)` | `AscendC::WholeReduceSum(...)` |  |
| `T.tile.wholereducemax(dst, src, ...)` | `AscendC::WholeReduceMax(...)` |  |
| `T.tile.wholereducemin(dst, src, ...)` | `AscendC::WholeReduceMin(...)` |  |

## MMA / GEMM

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `tl::ascend_mma(...)` | `AscendC::Mmad(...)` |  |
## 补充说明

| TileLang API | AscendC API | 备注 |
| --- | --- | --- |
| `T.copy(...)` | 不是单一 API | 会根据 scope 选择 `DataCopy` / `LoadData` / `Fixpipe` / `DataCopyPad` / `Cast`。 |
| `T.tile.add/sub/mul/div/max/min(...)` | 可能是向量版或标量版 | tensor-tensor 常对应 `Add/Sub/Mul/Div/Max/Min`；scalar 常对应 `Adds/Muls/Maxs/Mins`。 |
