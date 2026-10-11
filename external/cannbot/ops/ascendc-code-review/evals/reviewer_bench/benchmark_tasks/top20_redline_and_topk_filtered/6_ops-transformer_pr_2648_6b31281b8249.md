# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 2648
- **PR作者**: sangzhenguo
- **代码文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul.hpp, mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **代码侧别**: Kernel侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 11 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 11 | 6 | 100% |

---

## 发现问题


### 文件: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h（Kernel侧）

#### [1] 人工检视意见

- **提出人**: jiangbo96
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 168
- **评论时间**: 2026-03-17
- **Commit**: 6b31281b8249
- **问题描述**:

  > eventUbBiasVMTE2List未赋初始值即使用

- **代码片段**（行168）:
```cpp
 158 |             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
 159 |         }
 160 |     }
 161 | 
 162 |     // perChannel、perToken
 163 |     CATLASS_DEVICE
 164 |     void operator() (__gm__ ElementScale *ptrScale, LayoutScale layoutScale,
 165 |                      __gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
 166 |                      __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD, LayoutD layoutD,
 167 |                      GemmCoord problemShape)
 168 |     {
 169 |         // Calculate the offset of the current block
 170 |         MatrixCoord actualBlockShape = problemShape.GetCoordMN();
 171 | 
 172 |         gmScale.SetGlobalBuffer(ptrScale);
 173 |         gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
 174 |         gmC.SetGlobalBuffer(ptrC);
 175 |         gmD.SetGlobalBuffer(ptrD);
 176 | 
 177 |         auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
```

---

#### [2] 人工检视意见

- **提出人**: jiangbo96
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 178
- **评论时间**: 2026-03-17
- **Commit**: 6b31281b8249
- **问题描述**:

  > eventUbBiasVMTE2List同样未初始化赋值即使用

- **代码片段**（行178）:
```cpp
 168 |     {
 169 |         // Calculate the offset of the current block
 170 |         MatrixCoord actualBlockShape = problemShape.GetCoordMN();
 171 | 
 172 |         gmScale.SetGlobalBuffer(ptrScale);
 173 |         gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
 174 |         gmC.SetGlobalBuffer(ptrC);
 175 |         gmD.SetGlobalBuffer(ptrD);
 176 | 
 177 |         auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
 178 |         auto tileShape = TileShape::ToCoord();
 179 |         EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
 180 |         uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
 181 |         uint32_t subblockIdx = AscendC::GetSubBlockIdx();
 182 |         uint32_t subblockNum = AscendC::GetSubBlockNum();
 183 | 
 184 |         InitFlag();
 185 |         for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
 186 |             auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
 187 |             auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
```

---


#### [3] 人工检视意见

- **提出人**: jiangbo96
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 313
- **评论时间**: 2026-03-17
- **Commit**: 6b31281b8249
- **问题描述**:

  > 298行的PIPE_V可能还没做完，就执行了此时的CAST，会存在同步异常导致的精度问题。需要中间加上AscendC::PipeBarrier<PIPE_V>();

- **代码片段**（行313）:
```cpp
 303 |             auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
 304 |             auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();
 305 | 
 306 |             auto gmTileScale = gmScale[layoutScale.GetOffset(scaleTileOffset)];
 307 |             auto layoutGmTileScale = layoutScale.GetTileLayout(scaleTileShape);
 308 | 
 309 |             auto &ubScale = ubScaleList[ubListId];
 310 |             auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);
 311 | 
 312 |             // 把 scale 从GM拷贝到UB
 313 |             copyGmToUbScale(ubScale, gmTileScale, layoutUbScale, layoutGmTileScale);
 314 |             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
 315 | 
 316 |             // 在UB上把C cast到FP32
 317 |             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
 318 |             AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
 319 |             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
 320 | 
 321 |             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
 322 | 
```

---

#### [4] 人工检视意见

- **提出人**: jiangbo96
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 303
- **评论时间**: 2026-03-17
- **Commit**: 6b31281b8249
- **问题描述**:

  > 此处的ubBiasAdd赋值为ubMul，仅为vector上的UB操作，不需要和数据搬运流水存在时序关系。并且，可能存在SET/WAIT数量不配对的问题

- **代码片段**（行303）:
```cpp
 293 |             auto layoutGmTileC = layoutC.GetTileLayout(actualTileShape);
 294 | 
 295 |             auto &ubC = ubCList[ubListId];
 296 |             LayoutC layoutUbC{actualTileShape, ubTileStride};
 297 | 
 298 |             // 把 C 从GM拷贝到UB
 299 |             AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
 300 |             copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
 301 |             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
 302 | 
 303 |             auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
 304 |             auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();
 305 | 
 306 |             auto gmTileScale = gmScale[layoutScale.GetOffset(scaleTileOffset)];
 307 |             auto layoutGmTileScale = layoutScale.GetTileLayout(scaleTileShape);
 308 | 
 309 |             auto &ubScale = ubScaleList[ubListId];
 310 |             auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);
 311 | 
 312 |             // 把 scale 从GM拷贝到UB
```

---


#### [5] 人工检视意见

- **提出人**: liuboxi
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 135
- **评论时间**: 2026-03-20
- **Commit**: 6b31281b8249
- **问题描述**:

  > 功能问题，新增UB使用，需要保证不超UB大小。111行校验是否满足UB大小需要修改。

- **代码片段**（行135）:
```cpp
 125 |             ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
 126 |             ubOffset += TileShape::COUNT * sizeof(ElementD);
 127 | 
 128 |             eventUbCVMTE2List[i] = eventVMTE2++;
 129 |             eventUbCMTE2VList[i] = eventMTE2V++;
 130 |             eventUbScaleMTE2VList[i] = eventMTE2V++;
 131 |             eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
 132 |             eventUbDMTE3VList[i] = eventMTE3V++;
 133 |             eventUbDVMTE3List[i] = eventVMTE3++;
 134 |         }
 135 |         ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 136 |         ubOffset += TileShape::COUNT * sizeof(float);
 137 |         ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 138 |         ubOffset += TileShape::COUNT * sizeof(float);
 139 |         ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 140 |         ubOffset += TileShape::ROW * BYTE_PER_BLK;
 141 |         ubPerTokenMul = ubMul;
 142 |     }
 143 | 
 144 |     CATLASS_DEVICE
```

---

#### [6] 人工检视意见

- **提出人**: ranczhw
- **作者**: sangzhenguo
- **文件**: mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
- **行号**: 140
- **评论时间**: 2026-03-20
- **Commit**: 6b31281b8249
- **问题描述**:

  > 确保eventid没越界

- **代码片段**（行140）:
```cpp
 130 |             eventUbScaleMTE2VList[i] = eventMTE2V++;
 131 |             eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
 132 |             eventUbDMTE3VList[i] = eventMTE3V++;
 133 |             eventUbDVMTE3List[i] = eventVMTE3++;
 134 |         }
 135 |         ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 136 |         ubOffset += TileShape::COUNT * sizeof(float);
 137 |         ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 138 |         ubOffset += TileShape::COUNT * sizeof(float);
 139 |         ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
 140 |         ubOffset += TileShape::ROW * BYTE_PER_BLK;
 141 |         ubPerTokenMul = ubMul;
 142 |     }
 143 | 
 144 |     CATLASS_DEVICE
 145 |     void WaitFlag()
 146 |     {
 147 |         for (uint32_t i = 0; i < UB_STAGES; ++i) {
 148 |             AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
 149 |             AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
```

---


## 被检视代码

> 本报告基于 PR 2648 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `mc2/matmul_reduce_scatter_v2/op_kernel/matmul.hpp`
- `mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h`
