# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 5399
- **PR作者**: Caoyu_Jiang
- **代码文件**: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp, gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h, gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre_nz.h
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 14 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 14 | 12 | 100% |

---

## 发现问题

### 文件: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp（Tiling侧）


#### [1] 人工检视意见

- **提出人**: zhangshuoleiokok
- **作者**: Caoyu_Jiang
- **文件**: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- **行号**: 2119
- **评论时间**: 2026-05-16
- **Commit**: 4f464341248e
- **问题描述**:

  > 【review】此处出现魔鬼数字64 ，实际为NZ格式的int4数据类型对齐要求，建议修改为constexpr

- **代码片段**（行2119）:
```cpp
2109 |                     (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_ALLOW_WORKSPACE_INDEX] : 0;
2110 | 
2111 |           bool isWorkspaceValid = 
2112 |               (static_cast<int64_t>(preProcessWorkspaceSize) + workspaces[0] <= tuningConfigWorkspace_) || 
2113 |               (tuningConfigWorkspace_ == TUNING_WS_AUTO);
2114 |           printf("tuningConfigWorkspace_ = %lld\n", tuningConfigWorkspace_);
2115 |           // 空间足够 → 开启前处理 反之走原路径
2116 |           uint32_t isA8W4MSDPreNZ = 0;
2117 | 
2118 |           // workspace 合法, k64位对齐 → 开启 ND→NZ 预处理分支并且计算需要额外分配的空间大小
2119 |           if (isWorkspaceValid && k % 64 == 0) {
2120 |               isA8W4MSDPreNZ = 1;
2121 |               workspaces[0] += preProcessWorkspaceSize ;                
2122 |           }
2123 |           printf("workspaces[0] = %lld\n", workspaces[0]);
2124 |           printf("preProcessWorkspaceSize = %lld\n", preProcessWorkspaceSize);
2125 |           tilingDataA8W4.gmmBaseParams.set_isA8W4MSDPreNZ(isA8W4MSDPreNZ);
2126 |           printf("isA8W4MSDPreNZ = %u\n", isA8W4MSDPreNZ);
2127 |           tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
2128 |           context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());
```

---

#### [2] 人工检视意见

- **提出人**: zhangshuoleiokok
- **作者**: Caoyu_Jiang
- **文件**: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- **行号**: 2114
- **评论时间**: 2026-05-16
- **Commit**: 4f464341248e
- **问题描述**:

  > 【review】标准版本代码中禁止使用printf打印，容易造成误解且影响代码执行效率，此处应当删除

- **代码片段**（行2114）:
```cpp
2104 |           // --------------计算最小所需空间------------
2105 |           //矩阵空间
2106 |           uint64_t preProcessWorkspaceSize = (uint64_t)AlignUp(m, WS_ALIGN_8) * k * sizeof(int8_t) ;
2107 | 
2108 |           tuningConfigWorkspace_ = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_ALLOW_WORKSPACE_INDEX) ?
2109 |                     (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_ALLOW_WORKSPACE_INDEX] : 0;
2110 | 
2111 |           bool isWorkspaceValid = 
2112 |               (static_cast<int64_t>(preProcessWorkspaceSize) + workspaces[0] <= tuningConfigWorkspace_) || 
2113 |               (tuningConfigWorkspace_ == TUNING_WS_AUTO);
2114 |           printf("tuningConfigWorkspace_ = %lld\n", tuningConfigWorkspace_);
2115 |           // 空间足够 → 开启前处理 反之走原路径
2116 |           uint32_t isA8W4MSDPreNZ = 0;
2117 | 
2118 |           // workspace 合法, k64位对齐 → 开启 ND→NZ 预处理分支并且计算需要额外分配的空间大小
2119 |           if (isWorkspaceValid && k % 64 == 0) {
2120 |               isA8W4MSDPreNZ = 1;
2121 |               workspaces[0] += preProcessWorkspaceSize ;                
2122 |           }
2123 |           printf("workspaces[0] = %lld\n", workspaces[0]);
```

---

#### [3] 人工检视意见

- **提出人**: zhangshuoleiokok
- **作者**: Caoyu_Jiang
- **文件**: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- **行号**: 2201
- **评论时间**: 2026-05-16
- **Commit**: 4f464341248e
- **问题描述**:

  > 【review】无关调试代码禁止使用注释保留，此处应当删除

- **代码片段**（行2201）:
```cpp
2191 |   fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
2192 |   OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
2193 |   auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
2194 |   OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
2195 | 
2196 |   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
2197 | 
2198 |   compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
2199 |   // compileInfoPtr->aicNum = 1;
2200 |   compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
2201 |   // compileInfoPtr->aivNum = 2;
2202 |   compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
2203 |   compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
2204 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
2205 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
2206 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
2207 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
2208 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
2209 |   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
2210 | 
```

---

#### [4] 人工检视意见

- **提出人**: zhangshuoleiokok
- **作者**: Caoyu_Jiang
- **文件**: gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- **行号**: 2126
- **评论时间**: 2026-05-16
- **Commit**: 4f464341248e
- **问题描述**:

  > 【review】标准版本代码中禁止使用printf打印，容易造成客户误解且影响代码执行效率，此处应当删除

- **代码片段**（行2126）:
```cpp
2116 |           uint32_t isA8W4MSDPreNZ = 0;
2117 | 
2118 |           // workspace 合法, k64位对齐 → 开启 ND→NZ 预处理分支并且计算需要额外分配的空间大小
2119 |           if (isWorkspaceValid && k % 64 == 0) {
2120 |               isA8W4MSDPreNZ = 1;
2121 |               workspaces[0] += preProcessWorkspaceSize ;                
2122 |           }
2123 |           printf("workspaces[0] = %lld\n", workspaces[0]);
2124 |           printf("preProcessWorkspaceSize = %lld\n", preProcessWorkspaceSize);
2125 |           tilingDataA8W4.gmmBaseParams.set_isA8W4MSDPreNZ(isA8W4MSDPreNZ);
2126 |           printf("isA8W4MSDPreNZ = %u\n", isA8W4MSDPreNZ);
2127 |           tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
2128 |           context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());
2129 |           return ge::GRAPH_SUCCESS;
2130 |         }
2131 |       }
2132 | }
2133 | 
2134 | ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext* context) {
2135 |     OP_CHECK_NULL_WITH_CONTEXT(context, context);
```

---

#### [12] 人工检视意见

- **提出人**: Wei_NaChuan
- **作者**: Caoyu_Jiang
- **文件**: gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre_nz.h
- **行号**: 184
- **评论时间**: 2026-05-18
- **Commit**: 4f464341248e
- **问题描述**:

  > 【review】魔鬼数字64，修改掉

- **代码片段**（行184）:
```cpp
 174 |     PipeBarrier<PIPE_V>();
 175 |     Cast(xLowI4Tensor, xLowHalfTensor, AscendC::RoundMode::CAST_NONE, totalLen);
 176 |     PipeBarrier<PIPE_V>();
 177 |     auto tensorOut = vecOutQueue.AllocTensor<int4b_t>();
 178 |     
 179 |     //高四位低四位tensor交叉拼接
 180 |     DataCopyParams dataCopyParams{static_cast<uint16_t>(curVecBaseM), 1, static_cast<uint16_t>(curVecBaseK / 64 - 1), 1};
 181 |     auto loopCnt = curVecBaseK / 64;
 182 |     for(int i = 0; i < loopCnt; i++) {
 183 |         DataCopy(tensorOut[i * curVecBaseM * 2 * 64], xHighI4Tensor[i * 64],dataCopyParams);    //高四位
 184 |         DataCopy(tensorOut[64 + i * curVecBaseM * 2 * 64], xLowI4Tensor[i * 64],dataCopyParams);    //低四位，偏移量为64个元素
 185 |     }
 186 | 
 187 |     vecOutQueue.EnQue(tensorOut);
 188 | }
 189 | __aicore__ inline void GMMA8W4PreProcessNZ::copyOut(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM, uint32_t curVecBaseK)
 190 | {
 191 |     //m需要8位对齐
 192 |     auto m_Align_8 = AlignUp<8>(m);
 193 |     auto tensorOut = vecOutQueue.DeQue<int4b_t>();
```

---

## 被检视代码

> 本报告基于 PR 5399 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp`
- `gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h`
- `gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre_nz.h`
