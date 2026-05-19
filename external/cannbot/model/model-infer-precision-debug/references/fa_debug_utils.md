# FA 入参调试工具函数

在 FA 调用前插入 `debug_fa_all_params()` 一键校验全部入参，也可单独调用各子函数进行针对性检查。

## 函数索引

| 编号 | 函数 | 用途 |
|------|------|------|
| F.1 | `debug_atten_mask()` | atten_mask + sparse_mode 联合校验 |
| F.2 | `debug_fa_version_params()` | FA v1/v2 参数名混用检查 |
| F.3 | `debug_fa_layout_shape()` | input_layout 与 tensor shape 匹配 |
| F.4 | `debug_actual_seq_lengths()` | actual_seq_lengths 构造校验 |
| F.5 | `debug_inner_precise()` | inner_precise 行无效修正检查 |
| F.6 | `debug_mla_rope()` | MLA rope 参数校验 |
| F.7 | `debug_fa_all_params()` | 一键综合校验（调用 F.1-F.6） |

## 完整代码

使用时将以下代码复制到模型调试文件中，在 FA 调用前插入对应的 debug 函数。

```python
import torch


def debug_atten_mask(atten_mask, sparse_mode, q_seq_len, kv_seq_len, batch_size,
                     is_prefill, has_rope, layer_idx=0):
    """校验 atten_mask 与 sparse_mode 的配置正确性"""
    print(f"--- Layer {layer_idx} atten_mask Debug ---")
    print(f"  sparse_mode: {sparse_mode}")
    print(f"  is_prefill: {is_prefill}, q_seq_len: {q_seq_len}, kv_seq_len: {kv_seq_len}")

    if not is_prefill and sparse_mode != 0:
        if q_seq_len == 1 and not has_rope:
            print(f"  [WARN] Decode sparse_mode={sparse_mode}, Q_S=1 无 rope 时被忽略，"
                  f"但建议显式设为 0 避免歧义")
        else:
            print(f"  [ERROR] Decode 阶段 sparse_mode 应为 0, 当前为 {sparse_mode}")

    if not is_prefill and atten_mask is not None:
        print(f"  [WARN] Decode 阶段传入了 atten_mask (shape={atten_mask.shape})，"
              f"sparse_mode=0 时通常不需要 mask")

    if is_prefill and sparse_mode not in [2, 3]:
        print(f"  [WARN] Prefill 阶段 sparse_mode 通常为 2(leftUpCausal) 或 3(rightDownCausal), "
              f"当前为 {sparse_mode}")

    if atten_mask is None:
        if sparse_mode in [1, 2, 3, 4]:
            print(f"  [ERROR] sparse_mode={sparse_mode} 要求必须传入 atten_mask，但收到 None")
        else:
            print(f"  [OK] sparse_mode=0 + mask=None: 不做 mask 操作")
        return

    mask_shape = atten_mask.shape
    print(f"  mask shape: {mask_shape}, dtype: {atten_mask.dtype}")

    if atten_mask.dtype not in [torch.bool, torch.int8, torch.uint8]:
        print(f"  [ERROR] atten_mask dtype 必须为 bool/int8/uint8, 当前为 {atten_mask.dtype}")

    if sparse_mode in [2, 3, 4]:
        valid_shapes = [(2048, 2048), (1, 2048, 2048), (1, 1, 2048, 2048)]
        if mask_shape not in [torch.Size(s) for s in valid_shapes]:
            print(f"  [ERROR] sparse_mode={sparse_mode} 要求 mask shape 为 "
                  f"(2048,2048)/(1,2048,2048)/(1,1,2048,2048), 当前为 {mask_shape}")
        else:
            print(f"  [OK] mask shape 符合 sparse_mode={sparse_mode} 要求")
        if sparse_mode == 3 and mask_shape in [torch.Size(s) for s in valid_shapes]:
            mask_2d = atten_mask.view(2048, 2048)
            upper_tri = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)
            if atten_mask.dtype == torch.bool:
                if not mask_2d[upper_tri.bool()].all():
                    print(f"  [WARN] sparse_mode=3 的 mask 上三角部分未全部为 True")
    elif sparse_mode in [0, 1]:
        if len(mask_shape) >= 2:
            mask_q_s, mask_kv_s = mask_shape[-2:]
            if mask_q_s < q_seq_len:
                print(f"  [ERROR] mask Q_S 维度 ({mask_q_s}) 小于 query 实际长度 ({q_seq_len})")
            if mask_kv_s < kv_seq_len:
                print(f"  [ERROR] mask KV_S 维度 ({mask_kv_s}) 小于 KV 实际长度 ({kv_seq_len})")
    print(f"  [REMIND] 若使用 PA，mask 最后一维需 >= block_table 第二维 × block_size")


def debug_fa_version_params(fa_call_kwargs, fa_version):
    """检查 FA 调用参数是否与版本匹配"""
    v1_only_params = {"actual_seq_lengths", "scale", "num_heads",
                      "antiquant_scale", "antiquant_offset", "antiquant_mode",
                      "quant_scale1", "quant_scale2", "quant_offset2",
                      "dequant_scale1", "dequant_scale2", "softmax_lse_flag"}
    v2_only_params = {"actual_seq_qlen", "actual_seq_kvlen", "softmax_scale",
                      "num_query_heads", "dequant_scale_key", "dequant_scale_value",
                      "dequant_offset_key", "dequant_offset_value",
                      "key_quant_mode", "value_quant_mode",
                      "quant_scale_out", "quant_offset_out",
                      "return_softmax_lse", "learnable_sink", "dequant_scale_query"}

    wrong_version = v1_only_params if fa_version == "v2" else v2_only_params
    for param in fa_call_kwargs:
        if param in wrong_version:
            print(f"  [ERROR] 参数 '{param}' 属于 FA {'v1' if fa_version == 'v2' else 'v2'}，"
                  f"但当前使用 FA {fa_version}")

    if fa_version == "v2":
        if "softmax_scale" not in fa_call_kwargs:
            print(f"  [WARN] FA v2 未传 softmax_scale，将使用默认值 1.0")
        if "num_query_heads" not in fa_call_kwargs:
            print(f"  [WARN] FA v2 未传 num_query_heads，将使用默认值 1")
    elif fa_version == "v1":
        if "scale" not in fa_call_kwargs:
            print(f"  [WARN] FA v1 未传 scale，将使用默认值 1.0")
        if "num_heads" not in fa_call_kwargs:
            print(f"  [WARN] FA v1 未传 num_heads，将使用默认值 1")


def debug_fa_layout_shape(query, key, value, input_layout, num_query_heads,
                          num_kv_heads, block_table=None, layer_idx=0):
    """校验 input_layout 与 QKV tensor shape 的匹配性"""
    print(f"--- Layer {layer_idx} FA Layout Debug ---")
    print(f"  input_layout: {input_layout}")
    print(f"  Q: {query.shape}, K: {key.shape}, V: {value.shape}")
    q_shape = query.shape
    k_shape = key.shape
    layout_base = input_layout.split("_")[0]

    if layout_base == "BSH":
        if len(q_shape) != 3:
            print(f"  [ERROR] BSH layout 要求 Q 为 3D, 当前 {len(q_shape)}D")
    elif layout_base == "BNSD":
        if len(q_shape) != 4:
            print(f"  [ERROR] BNSD layout 要求 Q 为 4D, 当前 {len(q_shape)}D")
        elif q_shape[1] != num_query_heads:
            print(f"  [ERROR] BNSD Q N轴 ({q_shape[1]}) != num_query_heads ({num_query_heads})")
    elif layout_base == "TND":
        if len(q_shape) != 3:
            print(f"  [ERROR] TND layout 要求 Q 为 3D, 当前 {len(q_shape)}D")
        elif q_shape[1] != num_query_heads:
            print(f"  [ERROR] TND Q N轴 ({q_shape[1]}) != num_query_heads ({num_query_heads})")

    if block_table is not None:
        print(f"  [INFO] PA 模式, KV 按 block_table 索引")
        if block_table.ndim != 2:
            print(f"  [ERROR] block_table 必须为 2D, 当前 {block_table.ndim}D")


def debug_actual_seq_lengths(actual_seq_qlen, actual_seq_kvlen, input_layout,
                             is_prefill, batch_size, q_seq_len, kv_seq_len, layer_idx=0):
    """校验 actual_seq_lengths 的构造正确性"""
    layout_base = input_layout.split("_")[0]

    if layout_base == "TND":
        if actual_seq_qlen is None:
            print(f"  [ERROR] TND layout 必须传入 actual_seq_qlen")
            return
        qlen_list = actual_seq_qlen if isinstance(actual_seq_qlen, list) else actual_seq_qlen.tolist()
        for i in range(1, len(qlen_list)):
            if qlen_list[i] < qlen_list[i-1]:
                print(f"  [ERROR] TND actual_seq_qlen 必须单调递增 (cumsum), "
                      f"但 [{i-1}]={qlen_list[i-1]} > [{i}]={qlen_list[i]}")
    elif layout_base in ["BSH", "BNSD", "BSND"]:
        if actual_seq_kvlen is not None:
            kvlen_list = actual_seq_kvlen if isinstance(actual_seq_kvlen, list) else actual_seq_kvlen.tolist()
            for i, kv_l in enumerate(kvlen_list):
                if kv_l > kv_seq_len:
                    print(f"  [ERROR] actual_seq_kvlen[{i}]={kv_l} > KV S维度 ({kv_seq_len})")


def debug_inner_precise(atten_mask, inner_precise, q_seq_len, sparse_mode, layer_idx=0):
    """检查是否需要开启行无效修正"""
    if q_seq_len <= 1 or sparse_mode not in [0, 1] or atten_mask is None:
        return
    if atten_mask.dtype == torch.bool:
        all_masked_rows = atten_mask.all(dim=-1)
    else:
        all_masked_rows = (atten_mask != 0).all(dim=-1)
    num_invalid_rows = all_masked_rows.sum().item()
    if num_invalid_rows > 0 and inner_precise in [0, 1]:
        print(f"  [WARN] Layer {layer_idx}: {num_invalid_rows} 个全遮蔽行, "
              f"建议设置 inner_precise=2 或 3")


def debug_mla_rope(query, key, query_rope, key_rope, input_layout, sparse_mode, layer_idx=0):
    """校验 MLA rope 参数"""
    if (query_rope is None) != (key_rope is None):
        print(f"  [ERROR] query_rope 和 key_rope 必须同时配置或同时不配置")
        return
    if query_rope is None:
        return
    q_D = query.shape[-1]
    rope_D = query_rope.shape[-1]
    if q_D not in [512, 128]:
        print(f"  [ERROR] MLA query D 仅支持 512/128, 当前 {q_D}")
    if rope_D != 64:
        print(f"  [ERROR] rope D 必须为 64, 当前 {rope_D}")
    if q_D == 512 and sparse_mode not in [0, 3, 4]:
        print(f"  [ERROR] MLA D=512 仅支持 sparse_mode 0/3/4, 当前 {sparse_mode}")


def debug_fa_all_params(query, key, value, *, fa_version="v1", input_layout="BSH",
                        sparse_mode=0, atten_mask=None, actual_seq_qlen=None,
                        actual_seq_kvlen=None, num_heads=1, num_kv_heads=0, scale=1.0,
                        block_table=None, query_rope=None, key_rope=None,
                        inner_precise=0, is_prefill=True, layer_idx=0, **kwargs):
    """FA 入参一键综合校验"""
    print(f"\n{'='*60}")
    print(f"FA {fa_version} 综合校验 - Layer {layer_idx} ({'Prefill' if is_prefill else 'Decode'})")
    print(f"{'='*60}")

    batch_size = query.shape[0] if input_layout.startswith("B") else None
    q_seq_len = query.shape[1] if input_layout.startswith("B") else query.shape[0]
    kv_seq_len = key.shape[1] if len(key.shape) >= 3 else key.shape[0]

    if scale == 1.0:
        print(f"  [WARN] scale = 1.0, 确认是否忘记设置 1/sqrt(head_dim)?")

    has_rope = query_rope is not None
    debug_fa_layout_shape(query, key, value, input_layout, num_heads, num_kv_heads, block_table, layer_idx)
    debug_atten_mask(atten_mask, sparse_mode, q_seq_len, kv_seq_len, batch_size or 1, is_prefill, has_rope, layer_idx)
    debug_actual_seq_lengths(actual_seq_qlen, actual_seq_kvlen, input_layout, is_prefill, batch_size or 1, q_seq_len, kv_seq_len, layer_idx)
    debug_inner_precise(atten_mask, inner_precise, q_seq_len, sparse_mode, layer_idx)
    debug_mla_rope(query, key, query_rope, key_rope, input_layout, sparse_mode, layer_idx)
    print(f"{'='*60}\n")
```
