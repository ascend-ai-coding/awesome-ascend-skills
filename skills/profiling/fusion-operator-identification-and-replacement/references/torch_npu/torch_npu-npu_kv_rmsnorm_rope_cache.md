# torch\_npu.npu\_kv\_rmsnorm\_rope\_cache<a name="ZH-CN_TOPIC_0000002343094197"></a>

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

## 功能说明<a name="zh-cn_topic_0000002236535552_section1023311522369"></a>

- API功能：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。
    - **MLA（Multi-head Latent Attention）**：
      一种DeepSeek-V2提出的高效注意力机制，通过低秩潜在表示压缩 Key/Value，减少 KV Cache 和计算开销，同时保持长序列建模能力。

    - **RoPE（Rotary Position Embedding）**：
      一种位置编码方法，通过对向量施加旋转变换注入位置信息，能够建模相对位置关系，适用于长上下文场景，其中通过 RotateHalf 操作实现向量旋转。

    - **RMSNorm（Root Mean Square Normalization）**：
      一种归一化方法，通过均方根对输入进行缩放，计算简单、开销较低，常用于大模型训练。

- 计算公式：
    - **输入张量kv拆分**：拆分为两部分，其中B为批次大小，T为序列长度。以kv为576为例，拆分出512进行RMSNorm运算，拆分出64进行RoPE运算。

        $rms\_in \in \mathbb{R}^{B \times 1 \times T \times 512}$， $rope\_in \in \mathbb{R}^{B \times 1 \times T \times 64}$

    - **RMS归一化**：对rms\_in，应用RMS归一化。

        $\displaystyle y=\gamma\ \odot\ \left(\frac{rms\_in}{\sqrt{E_d[rms\_in^2]+\epsilon}}\right)$

        - $\gamma\in\mathbb{R}^{512}$是可学习的缩放参数。
        - $E_d[·]$表示沿最后一个维度（维度d=512）的均值。
        - $\epsilon$为小常数（如0.00001），防止除以零。
        - $\odot$表示逐元素相乘。

    - **旋转位置编码（RoPE）**

        - 重塑与转置：将rope_in重塑并转置以准备旋转。

          $k=reshape(rope\_in, [B, 1, T, 32, 2]) \rightarrow transpose(-1, -2) \rightarrow reshape([B, 1, T, 64])$

        - 旋转操作：应用旋转位置编码。

          $k_{embed}=k \odot cos + RotateHalf(k) \odot sin$

          - cos⁡和sin⁡为预计算的旋转角度参数。
          - RotateHalf(k)将k的后半部分元素移至前半部分并取反，后半部分用前半部分的值填充。具体来说，对于维度d=64：

            $RotateHalf(k)_i=-k_{i+32},\ if\ i < 32\ else\ k_{i-32}$

## 函数原型<a name="zh-cn_topic_0000002236535552_section123412524369"></a>

```python
torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002236535552_section1723416525369"></a>

- **kv** (`Tensor`)：必选参数，表示输入的特征张量。数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$BNSD$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, hidden\_size\]，其中hidden\_size=rms\_size\(RMS\)+rope\_size\(RoPE\)。
- **gamma** (`Tensor`)：必选参数，表示RMS归一化的缩放参数。数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。
- **cos** (`Tensor`)：必选参数，表示RoPE旋转位置编码的余弦分量。数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
- **sin** (`Tensor`)：必选参数，表示RoPE旋转位置编码的正弦分量。数据类型支持`torch.bfloat16`、`torch.float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
- **index** (`Tensor`)：必选参数，表示缓存索引张量，用于定位`k_cache`和`ckv_cache`的写入位置。数据类型支持`torch.int64`，数据格式为$ND$。shape取决于`cache_mode`。
- **k\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的k向量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`，数据格式为$ND$。shape取决于`cache_mode`。

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型不支持`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT系列产品</term>: 数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`
  <!-- end id5 -->

- **ckv\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的压缩后的kv向量。数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`，数据格式为$ND$。shape取决于`cache_mode`。

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型不支持`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`。
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>: 数据类型支持`torch.bfloat16`、`torch.float16`、`torch.int8`、`torch_npu.hifloat8`、`torch.float8_e5m2`、`torch.float8_e4m3fn`
  <!-- end id7 -->

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **k\_rope\_scale** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码的量化缩放因子。数据类型支持`torch.float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。量化模式下必填。
- **c\_kv\_scale** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化缩放因子。数据类型支持`torch.float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。量化模式下必填。
- **k\_rope\_offset** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码量化偏移量。数据类型支持`torch.float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：对称量化模式下该参数必填，暂不支持非对称量化。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持对称和非对称量化模式，非对称量化下该参数必填。
  <!-- end id9 -->

- **c\_kv\_offset** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化偏移量。数据类型支持`torch.float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：对称量化模式下该参数必填，暂不支持非对称量化。
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持对称和非对称量化模式，非对称量化下该参数必填。
  <!-- end id11 -->

- **epsilon** (`float`)：可选参数，默认值1e-5，表示RMS归一化中的极小值，防止除以零。
- **cache\_mode** (`str`)：可选参数，默认值'Norm'，表示缓存模式，支持的模式如下：

    | 枚举值 | 模式名 | 说明 |
    | --- | --- | --- |
    | Norm | KV-Cache更新模式 | k_cache形状为[batch_size, 1, cache_length, rope_size]，ckv_cache形状为[batch_size, 1, cache_length, rms_size]。index形状为[batch_size, seq_len]，index里的值表示每个Batch下的偏移。 |
    | PA/PA_BNSD | PagedAttention模式 | k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*seq_len]，index里的值表示每个token的偏移。 |
    | PA_NZ | Cache数据格式为FRACTAL_NZ的PagedAttention模式 | k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * seq_len]，index里的值表示每个token的偏移。<br>不同量化模式下数据排布不同：<br><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li> |
    | PA_BLK_BNSD | 特殊的PagedAttention模式 | k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*Ceil(seq_len/block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。 |
    | PA_BLK_NZ | Cache数据格式为FRACTAL_NZ的特殊的PagedAttention模式 | k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * Ceil(seq_len / block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。<br>不同量化模式下数据排布不同：<br><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li> |

- **is\_output\_kv** (`bool`)：可选参数，表示是否输出处理后的`k_embed_out`和`y_out`（未量化的原始值），默认值False不输出，仅`cache_mode`在\(PA/PA\_BNSD/PA\_NZ/PA\_BLK\_BNSD/PA\_BLK\_NZ\)模式下有效。

## 返回值说明<a name="zh-cn_topic_0000002236535552_section3234185215368"></a>

- **k\_cache** (`Tensor`)：和输入`k_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
- **ckv\_cache** (`Tensor`)：和输入`ckv_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
- **k\_embed\_out** (`Tensor`)：仅当`is_output_kv`为True时输出，表示RoPE处理后的值。要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]，数据类型和格式同输入`kv`一致。
- **y\_out** (`Tensor`)：仅当`is_output_kv`为True时输出，表示RMSNorm处理后的值。要求为4维张量，形状为\[batch\_size, 1, seq\_len, rms\_size\]，数据类型和格式同输入`kv`一致。

## 约束说明<a name="zh-cn_topic_0000002236535552_section1523425283618"></a>

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式。
- Tensor中shape使用的变量说明：
  - batch\_size：batch的大小。
  - seq\_len：sequence的长度。
  - hidden\_size：表示MLA输入的向量长度。
  - rms\_size：表示RMSNorm分支的向量长度。
  - rope\_size：表示RoPE分支的向量长度。
  - cache\_length：Norm模式下有效，表示KVCache支持的最大长度。
  - block\_num：PagedAttention模式下有效，表示Block的个数。
  - block\_size：PagedAttention模式下有效，表示Block的大小。

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：hidden\_size取值仅支持576，rms\_size取值仅支持512，rope\_size取值仅支持64。
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：hidden\_size、rms\_size、rope\_size由实际业务场景决定，用户按需设置，其中rope\_size必须为偶数，并且满足rms\_size+rope\_size=hidden\_size。
  <!-- end id13 -->

- 量化模式：当`k_rope_scale`和`c_kv_scale`非空时，`k_cache`和`ckv_cache`的dtype为`torch.int8`，缓存形状的最后一个维度需要为32（Cache数据格式为FRACTAL\_NZ模式），`k_rope_scale`和`c_kv_scale`必须同时非空。
- 非量化模式：当`k_rope_scale`和`c_kv_scale`为空时，`k_cache`和`ckv_cache`的dtype为`torch.bfloat16`或`torch.float16`。
- 非对称量化参数：`k_rope_offset`和`c_kv_offset`暂不支持配置。
- 索引映射：所有`cache_mode`缓存模式下，index的值不可以重复，如果传入的index值存在重复，算子的行为是未定义的且不可预知的。
    - Norm：index的值表示每个Batch下的偏移。
    - PA/PA\_BNSD/PA\_NZ：index的值表示全局的偏移。
    - PA\_BLK\_BNSD/PA\_BLK\_NZ：index的值表示每个页的全局偏移；这个场景假设cache更新是连续的，不支持非连续更新的cache。

- Shape关联规则：不同的`cache_mode`缓存模式有不同的Shape规则。
    - Norm：k\_cache形状为\[batch\_size, 1, cache\_length, rope\_size\]，ckv\_cache形状为\[batch\_size, 1, cache\_length, rms\_size\]，index形状为\[batch\_size, seq\_len\], cache\_length\>=seq\_len。
    - 非Norm模式\(PagedAttention相关模式\)：要求block\_num\>=Ceil\(seq\_len/block\_size\)\*batch\_size。

## 调用示例<a name="zh-cn_topic_0000002236535552_section3235105212365"></a>

- 单算子模式调用

    ```python
    # 示例1：基础样例（非量化，cache_mode="PA_BNSD"）
    import torch
    import torch_npu
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    
    # 示例2： 量化模式
    ## 1. 创建int8 cache (量化)
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. 准备量化参数 
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu() # K的RoPE部分scale
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu() # 压缩KV的scale
    # 其余代码不变

    # 示例3： Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # 其余代码不变

    # 示例4： PA_NZ
    cache_mode = "PA_NZ"
    # 其余代码不变

    # 示例5： PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # 其余代码不变

    # 示例6： PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # 其余代码不变
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16
    
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)

    # 示例2： 量化模式
    ## 1. 创建int8 cache (量化)
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. 准备量化参数 
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu() # K的RoPE部分scale
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu() # 压缩KV的scale
    # 其余代码不变

    # 示例3： Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # 其余代码不变

    # 示例4： PA_NZ
    cache_mode = "PA_NZ"
    # 其余代码不变

    # 示例5： PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # 其余代码不变

    # 示例5： PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # 其余代码不变
    ```
