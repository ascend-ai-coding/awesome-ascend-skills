# PyPTO API 详细索引

全部 API 文档。把 `<类>` 换成下方分区名（operation / tensor / config / datatype / symbolic / controlflow / element / others），`<name>` 换成该区下列的某个条目（已含 `pypto-` 前缀）。**缓存在场**读 `$PYPTO_DEVKIT_DIR/docs/api/<类>/<name>.md`；**无则在线** `https://pypto.gitcode.com/_sources/api/<类>/<name>.md.txt`。
例：tensor 区取 `pypto-Tensor-set_cache_policy` → 本地 `$PYPTO_DEVKIT_DIR/docs/api/tensor/pypto-Tensor-set_cache_policy.md` / 在线 `.../_sources/api/tensor/pypto-Tensor-set_cache_policy.md.txt`。
各区随版本增删，最新以缓存目录或 `_sources/api/<类>/index.md.txt` 为准。

## 目录
- operation 算子
- tensor 张量方法
- config 配置
- datatype 数据类型
- symbolic 符号/动态 shape
- controlflow 控制流
- element 逐元素
- others 其它/互转

## operation 算子
本地 `$PYPTO_DEVKIT_DIR/docs/api/operation/<name>.md` / 在线 `_sources/api/operation/<name>.md.txt` —— `<name>` 取下列之一：

pypto-abs, pypto-acos, pypto-acosh, pypto-add, pypto-amax, pypto-amin, pypto-arange, pypto-argmax, pypto-argmin, pypto-argsort, pypto-asin, pypto-asinh, pypto-assemble, pypto-atan, pypto-atan2, pypto-atanh, pypto-axpy_, pypto-bitwise_and, pypto-bitwise_left_shift, pypto-bitwise_not, pypto-bitwise_or, pypto-bitwise_right_shift, pypto-bitwise_xor, pypto-cast, pypto-cbrt, pypto-ceil, pypto-ceil_div, pypto-clip, pypto-clone, pypto-concat, pypto-conv, pypto-copysign, pypto-cos, pypto-cosh, pypto-cumprod, pypto-cumsum, pypto-deinterleave, pypto-dequantize, pypto-div, pypto-eq, pypto-erf, pypto-erfc, pypto-exp, pypto-exp2, pypto-expand_clone, pypto-expand_exp_dif, pypto-experimental-gather_in_l1, pypto-experimental-gather_in_ub, pypto-experimental-get_operation_options, pypto-experimental-online_softmax, pypto-experimental-online_softmax_update, pypto-experimental-set_operation_options, pypto-experimental-transposed_batchmatmul, pypto-expm1, pypto-fillpad, pypto-floor, pypto-floor_div, pypto-fmod, pypto-full, pypto-gather, pypto-gathermask, pypto-gcd, pypto-ge, pypto-gt, pypto-hypot, pypto-index_add, pypto-index_add_, pypto-index_add__ub, pypto-index_add_ub, pypto-index_put_, pypto-index_select, pypto-interleave, pypto-isfinite, pypto-le, pypto-log, pypto-log10, pypto-log1p, pypto-log2, pypto-logical_and, pypto-logical_not, pypto-lrelu, pypto-lt, pypto-matmul, pypto-maximum, pypto-minimum, pypto-mul, pypto-ne, pypto-neg, pypto-normal, pypto-one_hot, pypto-ones, pypto-pad, pypto-permute, pypto-pow, pypto-prelu, pypto-prod, pypto-quant_mx, pypto-quantize, pypto-reciprocal, pypto-relu, pypto-remainder, pypto-reshape, pypto-rms_norm, pypto-round, pypto-rsqrt, pypto-scaled_mm, pypto-scatter, pypto-scatter_, pypto-scatter_update, pypto-sigmoid, pypto-sign, pypto-signbit, pypto-sin, pypto-sinh, pypto-softmax, pypto-sqrt, pypto-sub, pypto-sum, pypto-tan, pypto-tanh, pypto-topk, pypto-transpose, pypto-tril, pypto-triu, pypto-trunc, pypto-uniform, pypto-unsqueeze, pypto-var, pypto-view, pypto-where, pypto-zeros


## tensor 张量方法
本地 `$PYPTO_DEVKIT_DIR/docs/api/tensor/<name>.md` / 在线 `_sources/api/tensor/<name>.md.txt` —— `<name>` 取下列之一：

pypto-Tensor_introduction, pypto-Tensor_constructor, pypto-Tensor-add, pypto-Tensor-amax, pypto-Tensor-amin, pypto-Tensor-assemble, pypto-Tensor-clone, pypto-Tensor-concat, pypto-Tensor-cos, pypto-Tensor-cumprod, pypto-Tensor-cumsum, pypto-Tensor-dim, pypto-Tensor-div, pypto-Tensor-dtype, pypto-Tensor-exp, pypto-Tensor-exp2, pypto-Tensor-expand_clone, pypto-Tensor-expm1, pypto-Tensor-fmod, pypto-Tensor-id, pypto-Tensor-index_add, pypto-Tensor-index_add_, pypto-Tensor-format, pypto-Tensor-gather, pypto-Tensor-gcd, pypto-Tensor-get_cache_policy, pypto-Tensor-log, pypto-Tensor-log10, pypto-Tensor-log2, pypto-Tensor-logical_not, pypto-Tensor-matmul, pypto-Tensor-maximum, pypto-Tensor-move, pypto-Tensor-mul, pypto-Tensor-name, pypto-Tensor-permute, pypto-Tensor-reciprocal, pypto-Tensor-remainder, pypto-Tensor-reshape, pypto-Tensor-round, pypto-Tensor-set_cache_policy, pypto-Tensor-scatter, pypto-Tensor-scatter_, pypto-Tensor-scatter_update, pypto-Tensor-shape, pypto-Tensor-rsqrt, pypto-Tensor-sigmoid, pypto-Tensor-sin, pypto-Tensor-softmax, pypto-Tensor-sqrt, pypto-Tensor-ceil, pypto-Tensor-floor, pypto-Tensor-trunc, pypto-Tensor-sub, pypto-Tensor-sum, pypto-Tensor-topk, pypto-Tensor-transpose, pypto-Tensor-tril, pypto-Tensor-tril_, pypto-Tensor-triu, pypto-Tensor-triu_, pypto-Tensor-unsqueeze, pypto-Tensor-view, pypto-Tensor-where, pypto-Tensor-getitem


## config 配置
本地 `$PYPTO_DEVKIT_DIR/docs/api/config/<name>.md` / 在线 `_sources/api/config/<name>.md.txt` —— `<name>` 取下列之一：

pypto-frontend-jit, pypto-get_codegen_options, pypto-get_cube_tile_shapes, pypto-get_debug_options, pypto-get_host_options, pypto-get_pass_options, pypto-get_pass_config, pypto-get_pass_configs, pypto-get_pass_default_config, pypto-get_vec_tile_shapes, pypto-get_verify_options, pypto-reset_options, pypto-set_codegen_options, pypto-set_cube_tile_shapes, pypto-set_debug_options, pypto-set_host_options, pypto-set_matrix_size, pypto-set_pass_config, pypto-set_pass_default_config, pypto-set_pass_options, pypto-set_semantic_label, pypto-set_vec_tile_shapes, pypto-set_verify_options


## datatype 数据类型
本地 `$PYPTO_DEVKIT_DIR/docs/api/datatype/<name>.md` / 在线 `_sources/api/datatype/<name>.md.txt` —— `<name>` 取下列之一：

CachePolicy, CastMode, DataType, PrecisionType, LogBaseType, OpType, OutType, ReduceMode, ReLuType, ScatterMode, SaturationMode, TileOpFormat, TopKAlgo, TransMode, pypto-bytes_of


## symbolic 符号/动态 shape
本地 `$PYPTO_DEVKIT_DIR/docs/api/symbolic/<name>.md` / 在线 `_sources/api/symbolic/<name>.md.txt` —— `<name>` 取下列之一：

pypto-SymbolicScalar_introduction, pypto-SymbolicScalar_constructor, pypto-SymbolicScalar-as_variable, pypto-SymbolicScalar-concrete, pypto-SymbolicScalar-is_concrete, pypto-SymbolicScalar-is_expression, pypto-SymbolicScalar-is_immediate, pypto-SymbolicScalar-is_symbol, pypto-SymbolicScalar-max, pypto-SymbolicScalar-min


## controlflow 控制流
本地 `$PYPTO_DEVKIT_DIR/docs/api/controlflow/<name>.md` / 在线 `_sources/api/controlflow/<name>.md.txt` —— `<name>` 取下列之一：

pypto-cond, pypto-function, pypto-is_loop_begin, pypto-is_loop_end, pypto-loop, pypto-loop_unroll


## element 逐元素
本地 `$PYPTO_DEVKIT_DIR/docs/api/element/<name>.md` / 在线 `_sources/api/element/<name>.md.txt` —— `<name>` 取下列之一：

pypto-Element_introduction, pypto-Element_constructor, pypto-Element-dtype, pypto-Element-value


## others 其它/互转
本地 `$PYPTO_DEVKIT_DIR/docs/api/others/<name>.md` / 在线 `_sources/api/others/<name>.md.txt` —— `<name>` 取下列之一：

pypto-from_torch, pypto-pass_verify_print, pypto-pass_verify_save, pypto-set_verify_golden_data

