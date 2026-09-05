# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 3540
- **PR作者**: number-2-bo-bo
- **代码文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md, attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md, attention/quant_lightning_indexer/examples/test_quant_lightning_indexer.cpp
- **代码侧别**: 通用
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 18 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 18 | 16 | 100% |

---

## 发现问题

### 文件: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md（通用）

#### [1] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 13
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > 这里是 KvQuantSparseFlashAttentionPioneer 算子的文档，但功能说明的开头写的是"QuantLightningIndexer是推理场景下"，把另一个算子的名称写进来了，需要改成 KvQuantSparseFlashAttentionPioneer 的描述。

- **代码片段**（行13）:
```markdown
   3 | [📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/kv_quant_sparse_flash_attention_pioneer)
   4 | 
   5 | ## 产品支持情况
   6 | 
   7 | |产品      | 是否支持 |
   8 | |:----------------------------|:-----------:|
   9 | |<term>Ascend 950PR/Ascend 950DT</term>|      √     |
  10 | 
  11 | ## 功能说明
  12 | 
  13 | - API功能：QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。引入param sink后，param sink加入到KV的首个基本快中进行Attention计算：
  14 | 
  15 | - 计算公式：
  16 |     $$
  17 |     \tilde{K}=\text{Gather}({Key};{SparseIndice})
  18 |     $$
  19 | 
  20 |     $$
  21 |     \tilde{K}_{\rm Nope}=\text{deQuant}_{8 \rightarrow 16}(\tilde{K}{...,:512};key\_scale)
  22 |     $$
```

---

#### [2] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 282
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > scaleValue 的描述写的是"用于标识输入query的量化模式"，这明显是 queryQuantMode 的描述。scaleValue 是一个 double 类型的缩放系数，描述应该说明它是 attention score 的缩放因子（通常是 1/sqrt(d_k)）。

- **代码片段**（行282）:
```markdown
 272 |       <td>表示添加在压缩value的序列维度上的额外参数。</td>
 273 |       <td>不支持空tensor。</td>
 274 |       <td>bfloat16、float16</td>
 275 |       <td>ND</td>
 276 |       <td>[sink_num, KV_N, D]</td>
 277 |       <td>x</td>
 278 |     </tr>
 279 |     <tr>
 280 |       <td>scaleValue</td>
 281 |       <td>输入</td>
 282 |       <td>用于标识输入`query`的量化模式。</td>
 283 |       <td>支持Per-Token-Head量化模式。</td>
 284 |       <td>INT64</td>
 285 |       <td>-</td>
 286 |       <td>-</td>
 287 |       <td>-</td>
 288 |     </tr>
 289 |     <tr>
 290 |       <td>keyQuantMode</td>
 291 |       <td>输入</td>
```

---

#### [3] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 284
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > scaleValue 的数据类型写的是 INT64，但函数原型里 scaleValue 的类型是 double，两边矛盾。这里应该改成 DOUBLE/FLOAT64。

- **代码片段**（行284）:
```markdown
 274 |       <td>bfloat16、float16</td>
 275 |       <td>ND</td>
 276 |       <td>[sink_num, KV_N, D]</td>
 277 |       <td>x</td>
 278 |     </tr>
 279 |     <tr>
 280 |       <td>scaleValue</td>
 281 |       <td>输入</td>
 282 |       <td>用于标识输入`query`的量化模式。</td>
 283 |       <td>支持Per-Token-Head量化模式。</td>
 284 |       <td>INT64</td>
 285 |       <td>-</td>
 286 |       <td>-</td>
 287 |       <td>-</td>
 288 |     </tr>
 289 |     <tr>
 290 |       <td>keyQuantMode</td>
 291 |       <td>输入</td>
 292 |       <td>用于标识输入key的量化模式。</td>
 293 |       <td>仅支持传入2，代表per_tile量化模式。</td>
```

---

#### [4] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 359
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > sparseMode 参数的数据类型写的是 INT32，但函数原型第64行 sparseMode 是 int64_t 类型。数据类型描述与函数签名不一致。

- **代码片段**（行359）:
```markdown
 349 |     <tr>
 350 |       <td>sparseMode</td>
 351 |       <td>输入</td>
 352 |       <td>表示sparse的模式。</td>
 353 |       <td>
 354 |           <ul>
 355 |                 <li>sparse_mode为0时，代表defaultMask模式。</li>
 356 |                 <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
 357 |           </ul>
 358 |       </td>
 359 |       <td>INT32</td>
 360 |       <td>-</td>
 361 |       <td>-</td>
 362 |       <td>-</td>
 363 |     </tr>
 364 |     <tr>
 365 |       <td>preTokens</td>
 366 |       <td>输入</td>
 367 |       <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
 368 |       <td>仅支持默认值2^63-1。</td>
```

---

#### [6] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 539
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > 约束说明中 query 的 D=576，key/value 的 D=656，但示例代码中 queryShape 和 keyShape 的最后一维都是 512。示例代码的 shape 不满足约束说明的要求，运行时可能出错。

- **代码片段**（行539）:
```markdown
 529 |   </table>
 530 | 
 531 | - **返回值：**
 532 | 
 533 |   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
 534 | 
 535 |   ## 约束说明
 536 | 
 537 | - 参数query中的N值为64/48/32/8/6/4/3/2，key、value的N支持1。
 538 | - 参数query中的D值为576，即nope+rope=512+64。
 539 | - 参数key、value中的D值为656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。
 540 | - 支持sparse_block_size整除block_size。
 541 | - 当前版本sink_num维仅支持128。
 542 | - layout_query为TND或BSND，sink场景layout_kv仅支持PA_BSND。
 543 | 
 544 | ## 调用示例
 545 | 
 546 | 示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
 547 | 
 548 | ```Cpp
```

---

#### [7] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 217
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > 括号不匹配，只有右括号"）"没有左括号。

- **代码片段**（行217）:
```markdown
 207 |       </td>
 208 |       <td>x</td>
 209 |     </tr>
 210 |     <tr>
 211 |       <td>blockTableOptional</td>
 212 |       <td>输入</td>
 213 |       <td>表示PageAttention中kvCache存储使用的block映射表。</td>
 214 |       <td>
 215 |           <ul>
 216 |                 <li>不支持空tensor。</li>
 217 |                 <li>PageAttention场景下，block_table必须为二维，第一维长度为B，第二维长度不小于所有batch中最大的s2对应的block数量，即s2_max / block_size向上取整）</li>
 218 |           </ul>
 219 |       </td>
 220 |       <td>INT32</td>
 221 |       <td>ND</td>
 222 |       <td>shape支持(B,S2/block_size)</td>
 223 |       <td>x</td>
 224 |     </tr>
 225 |     <tr>
 226 |       <td>actualSeqLengthsQueryOptional</td>
```

---

#### [8] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md
- **行号**: 538
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > 约束说明中 query 的 N 值列表是"64/48/32/8/6/4/3/2"，但参数说明第123行提到 TND 场景下"Q_N支持1/2/4/8/16/32/64/128"。两处 N 的支持范围不一致（约束里有 48/6/3 但没有 16/128），需要确认哪个是正确的。

- **代码片段**（行538）:
```markdown
 528 |   </tbody>
 529 |   </table>
 530 | 
 531 | - **返回值：**
 532 | 
 533 |   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
 534 | 
 535 |   ## 约束说明
 536 | 
 537 | - 参数query中的N值为64/48/32/8/6/4/3/2，key、value的N支持1。
 538 | - 参数query中的D值为576，即nope+rope=512+64。
 539 | - 参数key、value中的D值为656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。
 540 | - 支持sparse_block_size整除block_size。
 541 | - 当前版本sink_num维仅支持128。
 542 | - layout_query为TND或BSND，sink场景layout_kv仅支持PA_BSND。
 543 | 
 544 | ## 调用示例
 545 | 
 546 | 示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
 547 | 
```

---

### 文件: attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md（通用）


#### [10] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md
- **行号**: 130
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > key 参数的 shape 描述中，第129行用 layout_key，但第130-131行突然切换成了 layout_kv，同一个参数表格里两种写法混在一起。QuantLightningIndexer 的参数名是 layoutKeyOptional，应该统一用 layout_key。

- **代码片段**（行130）:
```markdown
 120 |       <td>
 121 |           <ul>
 122 |                 <li>Atlas A3 推理系列产品数据类型支持`int8`。</li>
 123 |                 <li>Ascend 950PR/Ascend 950DT数据类型支持`float8_e4m3fn、hifloat8`。</li>
 124 |           </ul>
 125 |       </td>
 126 |       <td>ND</td>
 127 |       <td>
 128 |           <ul>
 129 |                 <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
 130 |                 <li>layout_kv为BSND时，shape为(B, S2, N2, D)。</li>
 131 |                 <li>layout_kv为TND时，shape为(T2, N2, D)。</li>
 132 |           </ul>
 133 |       </td>
 134 |       <td>✓</td>
 135 |     </tr>
 136 |     <tr>
 137 |       <td>weights</td>
 138 |       <td>输入</td>
 139 |       <td>公式中的输入W。</td>
```

---

#### [11] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md
- **行号**: 295
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > sparseCount 参数的数据类型写的是 INT32，但函数原型里第43行 sparseCount 是 int64_t。需要改成 INT64。

- **代码片段**（行295）:
```markdown
 285 |       <td>STRING</td>
 286 |       <td>-</td>
 287 |       <td>-</td>
 288 |       <td>-</td>
 289 |     </tr>
 290 |     <tr>
 291 |       <td>sparseCount</td>
 292 |       <td>输入</td>
 293 |       <td>topK阶段需要保留的block数量。</td>
 294 |       <td>支持[1, 2048]</td>
 295 |       <td>INT32</td>
 296 |       <td>-</td>
 297 |       <td>-</td>
 298 |       <td>-</td>
 299 |     </tr>
 300 |     <tr>
 301 |       <td>sparseMode</td>
 302 |       <td>输入</td>
 303 |       <td>表示sparse的模式。</td>
 304 |       <td>
```

---

#### [12] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md
- **行号**: 448
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > 约束说明里写了"key、value的N支持1"，但 QuantLightningIndexer 的接口参数中没有 value 输入，这个约束不应该提 value。怀疑是从 KvQuantSparseFlashAttentionPioneer 那边复制过来时没改干净。

- **代码片段**（行448）:
```markdown
 438 |     </tr>
 439 |   </tbody>
 440 |   </table>
 441 | 
 442 | - **返回值：**
 443 | 
 444 |   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
 445 | 
 446 |   ## 约束说明
 447 | 
 448 | - 参数query中的N支持小于等于64/32/24/16，key、value的N支持1。
 449 | - headdim支持128。
 450 | - block_size取值为16的倍数，最大支持1024。
 451 | - 参数query、key的数据类型应保持一致。
 452 | - 对于Ascend 950PR/Ascend 950DT，当query和key的数据类型为`float8_e4m3fn`时，支持weights、query_dequant_scale、key_dequant_scale的数据类型为`bfloat16、float32、float32`或`float16、float16、float16`；当query和key的数据类型为`hifloat8`时，仅支持weights、query_dequant_scale、key_dequant_scale数据类型为`bfloat16、float32、float32`。
 453 | 
 454 | ## 调用示例
 455 | 
 456 | 示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
 457 | 
```

---

### 文件: attention/quant_lightning_indexer/examples/test_quant_lightning_indexer.cpp（通用）

#### [15] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/quant_lightning_indexer/examples/test_quant_lightning_indexer.cpp
- **行号**: 216
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > PrintOutResult 函数用 aclFloat16 类型接收输出结果，但 QuantLightningIndexer 的 out 是 INT32 类型（索引值），用 float16 去解读 int32 数据会得到乱码。应该用 int32_t 类型接收并打印。

- **代码片段**（行216）:
```cpp
 206 |     }
 207 | 
 208 |     ret = aclnnQuantLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
 209 |     if (!CHECK_RET(ret == ACL_SUCCESS)) {
 210 |         LOG_PRINT("aclnnQuantLightningIndexer failed. ERROR: %d\n", ret);
 211 |         return ret;
 212 |     }
 213 | 
 214 |     return ACL_SUCCESS;
 215 | }
 216 | 
 217 | int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
 218 |   auto size = GetShapeSize(shape);
 219 |   std::vector<aclFloat16> resultData(size, 0);
 220 |   auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
 221 |                          *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
 222 |   if (!CHECK_RET(ret == ACL_SUCCESS)) {
 223 |         LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
 224 |         return ret;
 225 |   }
```

---

#### [16] 人工检视意见

- **提出人**: miaofangzheng
- **作者**: number-2-bo-bo
- **文件**: attention/quant_lightning_indexer/examples/test_quant_lightning_indexer.cpp
- **行号**: 290
- **评论时间**: 2026-03-31
- **Commit**: 47af43bc865d
- **问题描述**:

  > main 里 PrintOutResult 用的 sparseIndicesShape 是 {1, 2, 1, 16}，最后一维是 16，但 InitializeTensors 里创建的 sparseIndicesShape 最后一维是 2048。两处 shape 不一致，打印结果时读取的元素个数和实际 tensor 大小不匹配。

- **代码片段**（行290）:
```cpp
 280 | }
 281 | 
 282 | } // namespace
 283 | 
 284 | int main() {
 285 |     int32_t deviceId = 0;
 286 |     aclrtStream stream = nullptr;
 287 |     TensorResources resources = {};
 288 |     void* workspaceAddr = nullptr;
 289 |     uint64_t workspaceSize = 0;
 290 |     std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 16};
 291 |     int ret = ACL_SUCCESS;
 292 | 
 293 |     // 1. Initialize device and stream
 294 |     ret = Init(deviceId, &stream);
 295 |     if (!CHECK_RET(ret == ACL_SUCCESS)) {
 296 |         LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
 297 |         return ret;
 298 |     }
 299 | 
```

---

## 被检视代码

> 本报告基于 PR 3540 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/kv_quant_sparse_flash_attention_pioneer/docs/aclnnKvQuantSparseFlashAttentionPioneer.md`
- `attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md`
- `attention/quant_lightning_indexer/examples/test_quant_lightning_indexer.cpp`
