# torch\_npu.npu\_fused\_causal\_conv1d

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能：**

    对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持 APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。

- **计算公式**：

  K是卷积核宽度（固定为3），L是原始序列长度，dim是特征维度。
  - 缓存读取

      缓存行索引：

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC 模式} \\
      cacheIndices[batchId], & \text{非 APC 且 cacheIndices 存在} \\
      batchId, & \text{其他}
      \end{cases}
      $$

      Case 1：首次计算（numComputedTokens[batchId] == 0）

      $$
      cachedState[i, dim] = 0, \quad 0 \leq i < K-1
      $$

      $$
      offset = 0
      $$

      Case 2：投机解码模式（numAcceptedTokens 存在）

      $$
      offset = numAcceptedTokens[batchId] - 1
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
      $$

      Case 3：默认模式

      $$
      offset = C - (K - 1)
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
      $$

  - 缓存拼接

      $$
      paddedInput[i, dim] =
      \begin{cases}
      cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
      x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
      \end{cases}
      $$

  - 缓存更新

      $$
      Len = offset + K - 1 + L
      $$

      $$
      M = \min(C, \; Len)
      $$

      $$
      writeCacheLine = \begin{cases}
      cacheIndices[batchId, \; idxLast], & \text{APC 模式} \\
      cacheIndices[batchId], & \text{非 APC 且 cacheIndices 存在} \\
      batchId, & \text{其他}
      \end{cases}
      $$

      $$
      convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
      $$

  - Offset 裁剪

      $$
      x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
      $$

  - APC 缓存填充（可选，APC 模式下）

      $$
      seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
      $$

      $$
      seqCompletedOffset = B - seqCompletedOffsetToken
      $$

      $$
      seqEndOffset = (L - seqCompletedOffset) \mod B
      $$

      $$
      lastFullBlockTokenIndex = \begin{cases}
      L - seqEndOffset - B, & seqEndOffset = 0 \\
      L - seqEndOffset, & \text{otherwise}
      \end{cases}
      $$

      $$
      nBlockToFill = idxLast - idxFirst
      $$

      对每个 chunk = 0, 1, ..., nBlockToFill - 1：

      $$
      boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
      $$

      $$
      convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
      $$

  - 因果1维卷积

      $$
      y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
      $$

  - 零填充重置（可选，当convMode == 1 并且 numComputedTokens不为空时）

      $$
      resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
      $$

      $$
      y[i, dim] = 0, \quad 0 \leq i < resetIdx
      $$

  - 残差连接（可选）

      $$
      y[i, dim] = x[i, dim] + y[i, dim]
      $$

## 函数原型

```python
torch_npu.npu_fused_causal_conv1d(x, weight, conv_states, *, query_start_loc=None, cache_indices=None,initial_state_mode=None, bias=None, num_accepted_tokens=None,activation="None", pad_slot_id=-1, run_mode=0, residual_connection=0, max_query_len=-1,num_computed_tokens=None, block_idx_first_scheduled_token=None,block_idx_last_scheduled_token=None, initial_state_idx=None, block_size=128, conv_mode="default") -> Tensor
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示输入序列，即公式中的$x$。数据类型支持`torch.float16`、`torch.bfloat16`，数据格式要求为ND，支持非连续的Tensor。不支持空Tensor。
- **weight**（`Tensor`）：必选参数，表示因果1维卷积核，即公式中的$weight$。数据类型、数据格式与`x`保持一致，不支持非连续的Tensor。不支持空Tensor。
- **conv\_states**（`Tensor`）：必选参数，表示缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新，即公式中的$conv\_states$。数据类型、数据格式与`x`保持一致，支持非连续的Tensor。不支持空Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **query\_start\_loc**（`Tensor`）：可选参数，表示序列起始位置索引，记录各序列在拼接张量`x`中的起始位置。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。当`x`的shape为2维时，不可省略，默认值为None。
- **cache\_indices**（`Tensor`）：可选参数，表示缓存索引，指定每个序列对应的缓存状态在conv\_states中的索引。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **initial\_state\_mode**（`Tensor`）：可选参数，遗留参数，该参数在当前接口内无作用。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **bias**（`Tensor`）：可选参数，遗留参数，该参数在当前接口内无作用。数据类型和数据格式与x保持一致，不支持非连续的Tensor。默认值为None。
- **num\_accepted\_tokens**（`Tensor`）：可选参数，表示每个batch的随机投机数。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **activation**（`str`）：可选参数，表示激活函数类型，该参数在当前接口内无作用。默认值为"None"。
- **pad\_slot\_id**（`int`）：可选参数，用于跳过不需要参与计算的batch，默认值为-1。
- **run\_mode**（`int`）：可选参数，该参数在当前接口内无作用。默认值为0。
- **residual\_connection**（`int`）：可选参数，用于判断输出结果是否要做残差连接。0：不做残差连接；1：输出为卷积结果与输入x之和（残差连接），默认值为0。
- **max\_query\_len**（`int`）：可选参数，表示所有batch中最大的seq_len，默认值为-1。
- **num\_computed\_tokens**（`Tensor`）：可选参数，当前batch已经处理的token总数，用于判断初始状态。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。conv_mode为“pangu”或者APC开启时，不可省略，默认值为None。
- **block\_idx\_first\_scheduled\_token**（`Tensor`）：可选参数，表示当前batch的第一个token对应的block索引。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。APC开启时，不可省略，默认值为None。
- **block\_idx\_last\_scheduled\_token**（`Tensor`）：可选参数，表示当前batch的最后一个token对应的block索引。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。APC开启时，不可省略，默认值为None。
- **initial\_state\_idx**（`Tensor`）：可选参数，表示初始索引块的索引。数据类型支持`torch.int32`，数据格式要求为ND，不支持非连续的Tensor。APC开启时，不可省略，默认值为None。
- **block\_size**（`int`）：可选参数，表示block块的大小，默认值为128。
- **conv\_mode**（`str`）：可选参数，支持Qwen3-Next("default")和Pangu V2("pangu")两种实现，默认值为"default"。

## 返回值说明

**y**（`Tensor`）：表示计算结果，公式中的$y$，数据类型和数据格式与`x`保持一致，不支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。

- 场景支持：
  - prefill场景：
    - x: [cu_seq_len, dim]
    - weight: [K, dim]，其中K=3
    - conv_states: [-1, K-1, dim]
    - query_start_loc: [batch+1]
    - cache_indices: 不开APC:[batch]或None, 开APC:[block, maxNumBlocks]
    - initial_state_mode: [batch]（无作用）
    - bias: [dim]（无作用）
    - num_accepted_tokens: [batch]（无作用）
    - num_computed_tokens: [batch]
    - block_idx_first_scheduled_token: 不开APC:None, 开APC:[batch]
    - block_idx_last_scheduled_token: 不开APC:None, 开APC:[batch]
    - initial_state_idx: 不开APC:None, 开APC:[batch]
    - activation: （无作用）
    - pad_slot_id: 默认值 -1
    - run_mode: （无作用）
    - max_query_len: 大于8
    - residual_connection: 不做残差: 0, 做残差：1
    - block_size: 典型值 128/256
    - conv_mode：Qwen3-Next模式: "default", Pangu V2: "pangu"
    - y: [cu_seq_len, dim]

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

    输入shape限制：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024\*1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64\*batch, 4G]，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。

  - prefill和decode混合场景：
    - x: [cu_seq_len, dim]
    - weight: [K, dim]，其中K=3
    - conv_states: [-1, K-1+m, dim]
    - query_start_loc: [batch+1]
    - cache_indices: 不开APC:[batch]或None, 开APC:[block, maxNumBlocks]
    - initial_state_mode: [batch]（无作用）
    - bias: [dim]（无作用）
    - num_accepted_tokens: [batch]
    - num_computed_tokens: [batch]
    - block_idx_first_scheduled_token: 不开APC:None, 开APC:[batch]
    - block_idx_last_scheduled_token: 不开APC:None, 开APC:[batch]
    - initial_state_idx: 不开APC:None, 开APC:[batch]
    - activation: （无作用）
    - pad_slot_id: 默认值 -1
    - run_mode: （无作用）
    - max_query_len: 大于8
    - residual_connection: 不做残差: 0, 做残差：1
    - block_size: 典型值 128/256
    - conv_mode：Qwen3-Next模式: "default", Pangu V2: "pangu"
    - y: [cu_seq_len, dim]

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

    输入shape限制：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024\*1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64\*batch, 4G]，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。

  - decode场景（变长序列）：
    - x: [cu_seq_len, dim]
    - weight: [K, dim]，其中K=3
    - conv_states: [-1, K-1+m, dim]
    - query_start_loc: [batch+1]
    - cache_indices: 不开APC:[batch]或None, 开APC:[block, maxNumBlocks]
    - initial_state_mode: [batch]（无作用）
    - bias: [dim]（无作用）
    - num_accepted_tokens: [batch]
    - num_computed_tokens: [batch]
    - block_idx_first_scheduled_token: 不开APC:None, 开APC:[batch]
    - block_idx_last_scheduled_token: 不开APC:None, 开APC:[batch]
    - initial_state_idx: 不开APC:None, 开APC:[batch]
    - activation: （无作用）
    - pad_slot_id: 默认值 -1
    - run_mode: （无作用）
    - max_query_len:默认值 1
    - residual_connection: 不做残差: 0, 做残差：1
    - block_size: 典型值 128/256
    - conv_mode：Qwen3-Next模式: "default", Pangu V2: "pangu"
    - y: [cu_seq_len, dim]

    其中state_len必须大于所有batch中最大的token个数加1。

    输入shape限制：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., k-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, batch*8]，每个batch的token个数范围为[1, 8]。dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。

  - decode场景（固定batch）：
    - x: [batch, m+1, dim]
    - weight: [K, dim]，其中K=3
    - conv_states: [-1, K-1+m, dim]
    - query_start_loc: [batch+1]
    - cache_indices: 不开APC:[batch]或None, 开APC:[block, maxNumBlocks]
    - initial_state_mode: [batch]（无作用）
    - bias: [dim]（无作用）
    - num_accepted_tokens: [batch]
    - num_computed_tokens: [batch]
    - block_idx_first_scheduled_token: 不开APC:None, 开APC:[batch]
    - block_idx_last_scheduled_token: 不开APC:None, 开APC:[batch]
    - initial_state_idx: 不开APC:None, 开APC:[batch]
    - activation: （无作用）
    - pad_slot_id: 默认值 -1
    - run_mode: （无作用）
    - max_query_len:默认值 1
    - residual_connection: 不做残差: 0, 做残差：1
    - block_size: 典型值 128/256
    - conv_mode：Qwen3-Next模式: "default", Pangu V2: "pangu"
    - y: [batch, m+1, dim]

    输入shape限制：
    - x支持3维[batch, m+1, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - m范围[0, 7]，dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。

- 输入值域限制：
  - query_start_loc是累计偏移量，取值范围[0, cu_seq_len]，长度为batch+1，query_start_loc[i]表示第i个序列的起始偏移，query_start_loc[batch+1]表示最后一个序列的结束位置。
  - blockSize 必须大于等于 2。
  - APC 开启时，必须提供 blockIdxFirstScheduledToken、blockIdxLastScheduledToken 、initialStateIdx和num_computed_tokens，且满足如下需求，i为batch的索引：
      - initialStateIdx[i] <= blockIdxFirstScheduledToken[i]+1
      - initialStateIdx[i] <= blockIdxLastScheduledToken[i]
      - blockIdxFirstScheduledToken[i] <= blockIdxLastScheduledToken[i]
      - blockIdxLastScheduledToken[i] < maxNumBlocks
  - num_accepted_tokens分为None和非None，非None情况下长度为batch，每个元素取值不超过当前batch的token数-1且大于0。
  - cache_indices的取值范围为[0, conv_states.dim[0]-1],且元素均不能相等。
  - Pangu V2 模式（conv_mode = "pangu"）下，num_computed_tokens不能为None。
  - 算子入参与中间计算结果，在对应数据类型（`torch.float16`/`torch.bfloat16`）下，数值均不会超出该类型值域范围。
  - 算子输入不支持有±inf和nan的情况。

## 调用示例

- 单算子模式调用
  - prefill场景：

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [15, 12, 20, 10]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 15, 27, 47, 57], dtype=torch.int32).npu()

    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([10, 5, 0, 0], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
      x,
      weight,
      conv_states,
      query_start_loc=query_start_loc,
      cache_indices=cache_indices,
      initial_state_mode=None,
      bias=None,
      num_accepted_tokens=None,
      num_computed_tokens=num_computed_tokens,
      block_idx_first_scheduled_token=None,
      block_idx_last_scheduled_token=None,
      initial_state_idx=None,
      activation="None",
      pad_slot_id=-1,
      run_mode=0,
      max_query_len=max_query_len,
      residual_connection=residual_connection,
      block_size=block_size,
      conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - prefill和decode混合场景：

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [12, 9, 2, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 12, 21, 23, 24], dtype=torch.int32).npu()

    num_slots = 8
    m = 2
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([0, 20, 50, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 2, 1], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - decode场景（变长序列）：

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()

    num_slots = 8
    m = 1
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 1, 1], dtype=torch.int32).npu()

    max_query_len = 1
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - decode场景（固定batch）：

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    m = 2
    seq_len = m + 1
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()

    num_slots = 8
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    query_start_loc = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()

    max_query_len = seq_len
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

- 图模式调用
  - prefill场景：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [15, 12, 20, 10]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 15, 27, 47, 57], dtype=torch.int32).npu()

    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([10, 5, 0, 0], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_prefill(x, weight, conv_states, query_start_loc,
                              cache_indices, num_computed_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=None,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_prefill, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens)
    print(f"output shape: {out.shape}")
    ```

  - prefill和decode场景：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [12, 9, 2, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 12, 21, 23, 24], dtype=torch.int32).npu()

    num_slots = 8
    m = 2
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([0, 20, 50, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 2, 1], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_mixed(x, weight, conv_states, query_start_loc,
                            cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_mixed, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```

  - decode场景（变长序列）：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()

    num_slots = 8
    m = 1
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 1, 1], dtype=torch.int32).npu()

    max_query_len = 1
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_decode_2d(x, weight, conv_states, query_start_loc,
                                cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_decode_2d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```

  - decode场景（固定batch）：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    m = 2
    seq_len = m + 1
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()

    num_slots = 8
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    query_start_loc = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()

    max_query_len = seq_len
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_decode_3d(x, weight, conv_states, query_start_loc,
                                cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_decode_3d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```
    