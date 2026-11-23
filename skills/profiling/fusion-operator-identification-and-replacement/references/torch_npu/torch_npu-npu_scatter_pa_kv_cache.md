# torch_npu.npu_scatter_pa_kv_cache

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- **API功能**：更新KvCache中指定位置的key和value。

- **输入输出支持场景**：

    - 场景一：

        ```python
        key: [batch *seq_len, num_head, k_head_size]
        value: [batch *seq_len, num_head, v_head_size]
        key_cache: [num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
        value_cache: [num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
        slot_mapping: [batch *seq_len]
        ```   

        其中`k_head_size`、`v_head_size`可以不同、也可以相同。

    - 场景二：

        ```python    
        key: [batch, seq_len, num_head, k_head_size]
        value: [batch, seq_len, num_head, v_head_size]
        key_cache: [num_blocks, block_size, 1, k_head_size]
        value_cache: [num_blocks, block_size, 1, k_head_size]
        slot_mapping: [batch, num_head]
        compress_lens: [batch, num_head]
        seq_lens: [batch]
        compress_seq_offsets: [batch * num_head]
        ```    

    上述场景根据构造的参数来区别，符合第一种入参构造走场景一，符合第二种构造走场景二。场景一没有`compress_lens`、`seq_lens`、`compress_seq_offsets`这三个可选参数。

## 函数原型

```python
torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, *, compress_lens = None, compress_seq_offsets = None, seq_lens = None, cache_mode = 'PA_NZ') -> ()
```

## 参数说明

- **key**（`Tensor`）：必选参数。表示待更新的key值，当前step多个token的`key`，支持3维或4维。数据类型支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`、`torch.uint8`、`torch.int16`、`torch.uint16`、`torch.int32`、`torch.uint32`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`，数据格式支持$ND$。
- **value**（`Tensor`）：必选参数，表示待更新的value值，当前step多个token的`value`，支持3维或4维，数据类型和数据格式与`key`保持一致。
- **key_cache**（`Tensor`）：必选参数，表示需要更新的`key_cache`，当前layer的`key_cache`，只支持4维，数据类型和数据格式与`key`保持一致。
- **value_cache**（`Tensor`）：必选参数，表示需要更新的`value_cache`，当前layer的`value_cache`。只支持4维，数据类型和数据格式与`key`保持一致。
- **slot_mapping**（`Tensor`）：必选参数，表示每个token key或value在cache中的存储偏移，数据类型支持`torch.int32`和`torch.int64`，数据格式支持$ND$。取值范围[0, num_blocks * block_size-1]，且元素值保证不重复，重复时不保证正确性。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **compress_lens**（`Tensor`）：可选参数，表示压缩量，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。
- **compress_seq_offsets**（`Tensor`）：可选参数，表示每个batch每个head的压缩起点，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。
- **seq_lens**（`Tensor`）：可选参数，表示每个batch的实际seqLens，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。
- **cache_mode**（`str`）：可选参数，表示`key_cache`和`value_cache`的内存排布格式。当传None或'Norm'时，仅支持ND内存排布格式。当传入'PA_NZ'时，仅支持NZ内存排布格式，默认值为'PA_NZ'。

## 返回值说明

无返回值，`key_cache`和`value_cache`会被原地更新。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- 输入参数不支持非连续；
- `key`、`value`、`key_cache`、`value_cache`的数据类型必须一致；
- `slot_mapping`、`compress_lens`、`compress_seq_offsets`、`seq_lens`的数据类型必须一致；
- `slot_mapping`的值范围[0, num_blocks * block_size-1]，且`slot_mapping`内的元素值保证不重复，重复时不保证正确性；
- 当`key`和`value`都是3维，则`key`和`value`的前两维`shape`必须相同；
- 当`key`和`value`都是4维，则`key`和`value`的前三维`shape`必须相同，且`key_cache`和`value_cache`的第三维必须是1；
- 当`key`和`value`是4维时，`compress_lens`、`seq_lens`为必选参数；当`key`和`value`是3维时，`compress_lens`、`compress_seq_offsets`、`seq_lens`为可选参数；
- 当`key`和`value`都是4维时，`slot_mapping`是二维，且`slot_mapping`的第一维值等于`key`的第一维为`batch`，`slot_mapping`的第二维值等于`key`的第三维为num_head(对应场景二)；
- 当`key`和`value`都是4维时，`seq_lens`是一维，且`seq_lens`的值等于`key`的第一维为batch(对应场景二)；
- `seq_lens`和`compress_lens`里面的每个元素值必须满足公式：reduceSum(seq_lens[i] - compress_lens[i]) <= num_blocks * block_size(对应场景二)；

## 调用示例

单算子模式调用：

- 场景一

    ```python
    import numpy as np
    import torch
    import torch_npu
    
    bs = 16
    num_head = 4
    k_head_size = 32
    v_head_size = 64
    num_blocks = 2
    lastDim_k = 16
    block_size = 32
    
    key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
    value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
    key_cache = np.random.randn(
        num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
    value_cache = np.zeros(
        (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
    slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)
    
    key_npu = torch.from_numpy(key).npu()
    value_npu = torch.from_numpy(value).npu()
    key_cache_npu = torch.from_numpy(key_cache).npu()
    value_cache_npu = torch.from_numpy(value_cache).npu()
    key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
    value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
    slot_mapping_npu = torch.from_numpy(slot_mapping).npu()
    
    torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu_cast, value_cache_npu_cast, slot_mapping_npu)
    print(key_cache_npu_cast)
    tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [[ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]],
        ...,
        [[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120],
        ...,
        [[ 0.1113,  0.6255,  0.7686,  ...,  0.0247,  0.2490,  0.6909],
        [ 0.4312,  0.7954,  0.7339,  ...,  0.1154,  0.6440,  0.3342],
        [ 0.9570,  0.2869,  0.6489,  ...,  0.7451,  0.0234,  0.8843]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120]]]]
        device='npu:0', dtype=torch.float16)
    ```

- 场景二

    ```python
    import torch
    import torch_npu
    a1 = torch.rand(2, 2, 3, 7).to(torch.float32).npu()     # key
    a2 = torch.rand(2, 2, 1, 7).to(torch.float32).npu()    # key_cache
    a3 = torch.tensor([[0, 3, 6], [9, 12, 15]]).to(torch.int32).npu()               # slot_mapping
    a4 = torch.rand(2, 2, 3, 7).to(torch.float32).npu()     # value
    a5 = torch.rand(2, 2, 1, 7).to(torch.float32).npu()    # value_cache
    a6 = torch.tensor([[1, 0, 0],[0, 1, 0]]).to(torch.int32).npu()                # compress_lens
    a7 = torch.tensor([0, 0, 1, 0, 1, 1]).to(torch.int32).npu()                  # compress_seq_offsets
    a8 = torch.tensor([2,1]).to(torch.int32).npu()                   # seq_lens
    res = torch_npu.npu_scatter_pa_kv_cache(a1, a4, a2, a5, a3, compress_lens=a6,compress_seq_offsets=a7, seq_lens=a8, cache_mode="Norm")
    ```

图模式调用：

- 场景一

    ```python
    import numpy as np
    import torch
    import torch_npu

    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.aoe_config.aoe_mode = "2"
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # 数据生成
    bs = 16
    num_head = 4
    k_head_size = 32
    v_head_size = 64
    num_blocks = 2
    lastDim_k = 16
    block_size = 32

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, key, value, slot_mapping, key_cache, value_cache):
            torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)

    if __name__ == "__main__":
        torch_npu.npu.set_device(0)

        key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
        value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
        key_cache = np.random.randn(
            num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
        value_cache = np.zeros(
            (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
        slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

        key_npu = torch.from_numpy(key).npu()
        value_npu = torch.from_numpy(value).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
        value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
        slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        model = Model().npu()
        # 图模式调用
        model = torch.compile(model, backend=npu_backend, dynamic = False, fullgraph = True)
        model(key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast)

        # 这里的输出与单算子一致
    ```

- 场景二

    ```python
    import os
    import torch
    import torch_npu
    import torchair

    config = torchair.CompilerConfig()
    # 设置导出图中的Aten算子信息
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = torchair.get_npu_backend(compiler_config = config)

    def generate_input_data():
      a1 = torch.rand(2, 2, 3, 7).to(torch.float32)     # key
      a2 = torch.rand(2, 2, 1, 7).to(torch.float32)    # key_cache
      a3 = torch.tensor([[0, 3, 6], [9, 12, 15]]).to(torch.int32)               # slot_mapping
      a4 = torch.rand(2, 2, 3, 7).to(torch.float32)     # value
      a5 = torch.rand(2, 2, 1, 7).to(torch.float32)    # value_cache
      a6 = torch.tensor([[1, 0, 0],[0, 1, 0]]).to(torch.int32)                # compress_lens
      a7 = torch.tensor([0, 0, 1, 0, 1, 1]).to(torch.int32)                   # compress_seq_offset
      a8 = torch.tensor([2,1]).to(torch.int32)                            # seq_lens

      input_data = [a1, a2, a3, a4, a5, a6, a7, a8]
      return input_data

    class Model(torch.nn.Module):
      def __init__(self):
        super().__init__()
      def forward(self, key, key_cache, slot_mapping, value, value_cache, compress_lens = None, compress_seq_offsets = None, seq_lens = None):
        return torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, compress_lens = compress_lens, compress_seq_offsets = compress_seq_offsets, seq_lens = seq_lens, cache_mode = "Norm")

    input_data = generate_input_data()
    a1_npu = input_data[0].npu()
    a2_npu = input_data[1].npu()
    a3_npu = input_data[2].npu()
    a4_npu = input_data[3].npu()
    a5_npu = input_data[4].npu()
    a6_npu = input_data[5].npu()
    a7_npu = input_data[6].npu()
    a8_npu = input_data[7].npu()

    model = Model()
    model=model.to('npu')

    model = torch.compile(model, backend=npu_backend, dynamic=True)
    res = model(a1_npu, a2_npu, a3_npu, a4_npu, a5_npu, a6_npu, a7_npu, a8_npu)
    print('************** test npu_scatter_pa_kv_cache dtype is: ', a1_npu.dtype, a3_npu.dtype)

    print("key_cache==========",a2_npu)
    print("value_cache=====:",a5_npu)
    print("success")
    ```
