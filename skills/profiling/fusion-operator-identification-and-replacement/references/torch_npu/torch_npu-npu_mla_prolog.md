# torch_npu.npu_mla_prolog

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- API功能：推理场景下，Multi-Head Latent Attention（MLA）前处理的计算操作。

  主要计算过程分为四路：

  - 首先对输入x乘以W<sup>DQ</sup>进行下采样和RmsNorm后分成两路：第一路乘以W<sup>UQ</sup>和W<sup>UK</sup>经过两次上采样后得到q<sup>N</sup>；第二路乘以W<sup>QR</sup>后经过旋转位置编码（ROPE）得到q<sup>R</sup>。
  - 第三路是将输入x乘以W<sup>DKV</sup>进行下采样和RmsNorm后传入Cache中得到k<sup>C</sup>；
  - 第四路是输入x乘以W<sup>KR</sup>后经过旋转位置编码后传入另一个Cache中得到k<sup>R</sup>。
  
- 计算公式：
    - RmsNorm公式：
        $$RmsNorm(x) = \gamma \cdot \frac{x} {\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}}$$

    - Query计算公式：
        $$c^Q = RmsNorm(x \cdot W^{DQ})$$
        $$q^C = c^Q \cdot W^{UQ}$$
        $$q^N = q^C \cdot W^{UK}$$

    - Query ROPE旋转位置编码:
        $$q^R = ROPE(c^Q \cdot W^{QR})$$

    - Key计算公式:
        $$k^C = Cache(RmsNorm(x \cdot W^{DKV}))$$

    - Key ROPE旋转位置编码:
        $$k^R = Cache(ROPE(x \cdot W^{KR}))$$

## 函数原型

```python
torch_npu.npu_mla_prolog(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **token\_x**（`Tensor`）：必选参数，对应公式中x。shape支持2维和3维，格式为\(T, He\)和\(B, S, He\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **weight\_dq**（`Tensor`）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为\(He, Hcq\)，dtype支持`torch.bfloat16`，数据格式支持FRACTAL\_NZ（可通过`torch_npu.npu_format_cast`将ND格式转为FRACTAL\_NZ格式）。
- **weight\_uq\_qr**（`Tensor`）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为\(Hcq, N\*\(D+Dr\)\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持FRACTAL\_NZ。
    - 当`weight_uq_qr`为`torch.int8`类型时，`weight_uq_qr`是一个pertensor的量化后的输入，表示当前为部分量化场景。
        - 若`kv_cache`、`kr_cache`为`torch.bfloat16`类型，对应`kv_cache_out`、`kr_cache_out`为非量化输出，此时`dequant_scale_w_uq_qr`字段必须传入，`smooth_scales_cq`字段可选传入。
        - 若`kv_cache`、`kr_cache`为`torch.int8`类型，对应`kv_cache_out`、`kr_cache_out`为量化输出，此时`dequant_scale_w_uq_qr`、`quant_scale_ckv`、`quant_scale_ckr`字段必须传入，`smooth_scales_cq`字段可选传入。
    - 当`weight_uq_qr`为`torch.bfloat16`类型时，表示当前为非量化场景。
        此时`dequant_scale_w_uq_qr`、`quant_scale_ckv`、`quant_scale_ckr`、`smooth_scales_cq`字段不能传入（即为none）。
  
- **weight\_uk**（`Tensor`）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为\(N, D, Hckv\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **weight\_dkv\_kr**（`Tensor`）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为\(He, Hckv+Dr\)，dtype支持`torch.bfloat16`，数据格式支持FRACTAL\_NZ。
- **rmsnorm\_gamma\_cq**（`Tensor`）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hcq,\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rmsnorm\_gamma\_ckv**（`Tensor`）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为\(Hckv,\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rope\_sin**（`Tensor`）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **rope\_cos**（`Tensor`）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为\(T, Dr\)和\(B, S, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **cache\_index**（`Tensor`）：必选参数，表示用于存储`kv_cache`和`kr_cache`的索引。shape支持1维和2维，格式为\(T,\)和\(B, S\)，dtype支持`torch.int64`，数据格式支持ND。`cache_index`的取值范围为[0,BlockNum*BlockSize)，当前不会对`cache_index`传入值的合法性进行校验，需用户自行保证。
- **kv\_cache**（`Tensor`）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **kr\_cache**（`Tensor`）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **dequant\_scale\_x**（`Tensor`）：预留参数，暂未使用，使用默认值即可。
- **dequant\_scale\_w\_dq**（`Tensor`）：预留参数，暂未使用，使用默认值即可。
- **dequant\_scale\_w\_uq\_qr**（`Tensor`）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化方式为perchannel。shape支持2维，格式为\(1, N\*\(D+Dr\)\)，dtype支持`torch.float`，数据格式支持ND。
- **dequant\_scale\_w\_dkv\_kr**（`Tensor`）：预留参数，暂未使用，使用默认值即可。
- **quant\_scale\_ckv**（`Tensor`）：可选参数，用于对输出到`kv_cache_out`中的数据做量化操作时的参数。shape支持2维，格式为\(1, Hckv\)，dtype支持`torch.float`，数据格式支持ND。
- **quant\_scale\_ckr**（`Tensor`）：可选参数，用于对输出到`kr_cache_out`中的数据做量化操作时的参数。shape支持2维，格式为\(1, Dr\)，dtype支持`torch.float`，数据格式支持ND。
- **smooth\_scales\_cq**（`Tensor`）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为\(1, Hcq\)，dtype支持`torch.float`，数据格式支持ND。
- **rmsnorm\_epsilon\_cq**（`float`）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，默认值为1e-05。
- **rmsnorm\_epsilon\_ckv**（`float`）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，默认值为1e-05。
- **cache\_mode**（`str`）：可选参数，表示`kv_cache`的模式，支持`PA_BSND`、`PA_NZ`，默认值为`PA_BSND`。

## 返回值说明

- **query**（`Tensor`）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为\(T, N, Hckv\)和\(B, S, N, Hckv\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **query\_rope**（`Tensor`）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为\(T, N, Dr\)和\(B, S, N, Dr\)，dtype支持`torch.bfloat16`，数据格式支持ND。
- **kv\_cache\_out**（`Tensor`）：表示Key输出到`kv_cache`中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Hckv\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。
- **kr\_cache\_out**（`Tensor`）：表示Key的位置编码输出到`kr_cache`中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为\(BlockNum, BlockSize, Nkv, Dr\)，dtype支持`torch.bfloat16`和`torch.int8`，数据格式支持ND。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 接口参数中shape格式字段含义：
    - B：Batch表示输入样本批量大小，取值范围为0\~65536。
    - S：Seq-Length表示输入样本序列长度，取值范围为0\~16。
    - He：Head-Size表示隐藏层的大小，取值为7168。

    - Hcq：q低秩矩阵维度，取值为1536。
    - N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    - Hckv：kv低秩矩阵维度，取值为512。
    - D：qk不含位置编码维度，取值为128。
    - Dr：qk位置编码维度，取值为64。
    - Nkv：kv的head数，取值为1。
    - BlockNum：PagedAttention场景下的块数，取值为计算B\*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
    - BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    - T：BS合轴后的大小，取值范围：0\~1048576。注：若采用BS合轴，此时token\_x、rope\_sin、rope\_cos均为2维，cache\_index为1维，query、query\_rope为3维。
- shape约束
    - B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        - 如果Skv取值为0，则query、query_rope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import math
    # 生成随机数据, 并发送到npu
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # 调用MlaProlog算子
    query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    print(query_mla)
    # 执行上述代码的输出out类似如下
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import math
    import torchair as tng
    
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    import os
    os.environ["TORCHDYNAMO_VERBOSE"]="1"
    os.environ["TORCH_LOGS"]="+dynamo"
    
    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # 数据生成
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
        print("single op output:", query_mla)
        print("graph output:", graph_output)
        
    if __name__ == "__main__":
        MetaInfershape()
    
    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    
    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16) 
    ```
