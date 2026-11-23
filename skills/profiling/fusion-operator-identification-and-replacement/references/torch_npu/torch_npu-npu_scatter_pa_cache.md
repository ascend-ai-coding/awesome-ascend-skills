# torch_npu.npu_scatter_pa_cache

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- **API功能**：更新KCache中指定位置的key。

- **计算公式**：

  根据可选参数组合，可以分为如下三个场景：

  - 场景一：未使用compress\_lens、seq\_lens、compress\_seq\_offset参数

    $$\mathit{keyCache} = \mathit{slotMapping}(\mathit{key})$$

  - 场景二：

    $$
    \begin{aligned}
    \mathit{keyCache} &= \mathit{slotMapping}(\mathit{key}[: \mathit{compressSeqOffset}], \\
    &\quad \mathit{ReduceMean}(\mathit{key}[\mathit{compressSeqOffset}: \mathit{compressSeqOffset} + \mathit{compressLens}]), \\
    &\quad \text{key}[\text{compressSeqOffset} + \mathit{compressLens}: \mathit{seqLens}])
    \end{aligned}
    $$

  - 场景三：未使用compress\_seq\_offset参数

    $$\mathit{keyCache} = \mathit{slotMapping}(\mathit{key}[\mathit{seqLens} - \mathit{compressLens}: \mathit{seqLens}])$$

## 函数原型

```python
torch_npu.npu_scatter_pa_cache(key, slot_mapping, *, compress_lens = None, compress_seq_offset = None, seq_lens = None, key_cache) -> ()
```

## 参数说明

- **key**（`Tensor`）：必选参数，待更新的key值，数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`、`torch.uint8`、`torch.int16`、`torch.uint16`、`torch.int32`、`torch.uint32`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1`、`torch_npu.float4_e1m2`，tensor支持3-4维。其中数据类型`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`、`torch_npu.float4_e2m1`、`torch_npu.float4_e1m2`仅当tensor为3维的时候支持，shape为\(batch \* seq\_len, num\_head, k\_head\_size\)或\(batch, seq\_len, num\_head, k\_head\_size\)，并且`torch_npu.float4_e2m1`、`torch_npu.float4_e1m2`情况下，k\_head\_size必须是偶数。数据格式支持$ND$。
- **slot\_mapping**（`Tensor`）：必选参数，key的每个token在cache中的存储偏移，数据类型支持`torch.int32`、`torch.int64`，tensor支持1-2维，当key是3维时，slot\_mapping shape为\(batch \* seq\_len\)；当key是4维时，slot\_mapping shape为\(batch, num\_head\)。值范围为\[0, num\_blocks \* block\_size - 1\]，且元素值不能重复，重复时不保证正确性。数据格式支持$ND$。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **compress\_lens**（`Tensor`）：可选参数，表示压缩量，数据类型与`slot_mapping`一致，tensor支持1-2维，当key是4维且compress\_seq\_offset不为None时，shape为\(batch, num\_head\)，当key是4维且compress\_seq\_offset为None时，shape为\(batch \* num\_head\)。数据格式支持$ND$。
- **compress\_seq\_offset**（`Tensor`）：可选参数，表示每个batch中每个head的压缩起点，数据类型与`slot_mapping`一致，tensor支持1维，shape为\(batch \* num\_head\)。数据格式支持$ND$。
- **seq\_lens**（`Tensor`）：可选参数，表示每个batch的实际seqLens，数据类型与`slot_mapping`一致，tensor支持1维，shape为\(batch\)。数据格式支持$ND$。
- **key\_cache**（`Tensor`）：必选参数，输出张量，需要更新的`key_cache`，数据类型与`key`一致。当key是3维时，shape为\(num\_blocks, block\_size, num\_head, k\_head\_size\)，当key是4维时，shape为\(num\_blocks, block\_size, 1, k\_head\_size\)。数据格式支持$ND$。

## 返回值说明

无返回值，直接修改入参key\_cache。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口仅支持单算子模式。
- 不支持非连续的Tensor。支持空Tensor。
- `seq_lens`和`compress_lens`里面的每个元素值必须满足公式：reduceSum\(seq\_lens\[i\] - compress\_lens\[i\] + 1\) <= num\_blocks \* block\_size（对应场景二、三）。
- 参数说明里Shape使用的变量说明：
  - batch：当前输入的序列数量（一次处理的样本数），取值为正整数。
  - seq\_len：序列的长度，取值为正整数。
  - num\_head：多头注意力中“头”的数量，取值为正整数。
  - k\_head\_size：每个注意力头中key的特征维度（单头key的长度），取值为正整数。
  - num\_blocks：keyCache中预分配的块总数，用于存储所有序列的key数据，取值为正整数。
  - block\_size：每个缓存块包含的token数量，取值为正整数。

## 调用示例

单算子模式调用：

- 场景一

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (256, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(256).to(torch.int32).npu() # slot_mapping
    torch_npu.npu_scatter_pa_cache(a1, a3, key_cache=a2)
    ```

- 场景二

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 256, 1, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(16, 16).to(torch.int32).npu() # slot_mapping
    a4 = torch.randint(0, 8, (16, 16), dtype=torch.int32).npu() # compress_lens
    a5 = torch.randint(0, 8, (256,), dtype=torch.int32).npu() # compress_seq_offsets
    a6 = torch.full((16,), 16).to(torch.int32).npu() # seq_len
    torch_npu.npu_scatter_pa_cache(a1, a3, compress_lens=a4, compress_seq_offsets=a5, seq_lens=a6, key_cache=a2)
    ```

- 场景三

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 256, 1, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(16, 16).to(torch.int32).npu() # slot_mapping
    a4 = torch.randint(0, 8, (256,), dtype=torch.int32).npu() # compress_lens
    a6 = torch.full((16,), 16).to(torch.int32).npu() # seq_len
    torch_npu.npu_scatter_pa_cache(a1, a3, compress_lens=a4, seq_lens=a6, key_cache=a2)
    ```
