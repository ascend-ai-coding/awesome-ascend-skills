# PyPTO DFX 经验表

> 本表是主表，收录已验证的修复配方。未命中时查 [问题查找表](problem-lookup.md)（辅助表，路由到官方文档）。

---

## 错误码/关键词导航索引

> 根据报错中的**错误码**或**关键词**快速定位到对应分类文件。

| 错误码 / 关键词 | 跳转 |
|----------------|------|
| `F00001` + `Tensor is not iterable` | [external §1](experience_classified/external_error.md) |
| `F00001` + `not a valid torch tensor type` / `from_torch()` 返回值传给 `@jit` kernel | [external §1](experience_classified/external_error.md) |
| `F00001` + `Element(): incompatible constructor` | [external §2](experience_classified/external_error.md) |
| `F00001` + `RecordIfBranch` / `pypto.cond` 不接受任意比较表达式 | [external §3](experience_classified/external_error.md) |
| `F00005` + `With is not supported` / `pypto.cond` 不支持 `with` 语法 | [external §3](experience_classified/external_error.md) |
| `F00001` + `Return statements are not allowed` | [external §4](experience_classified/external_error.md) |
| `F00001` + `reshape() requires integer shape` + `np.int64` | [external §4](experience_classified/external_error.md) |
| `F00001` + `pypto.zeros` dtype 被 `*size` 吞掉 | [external §5](experience_classified/external_error.md) |
| `F00001` + `Div(): incompatible function arguments` | [external §6](experience_classified/external_error.md) |
| `F00001` + `topk() got an unexpected keyword argument 'sorted'` | [external §14](experience_classified/external_error.md) |
| `F00001` + `rms_norm() got an unexpected keyword argument 'eps'` / `epsilon` 参数名 | [external §17](experience_classified/external_error.md) |
| `F00001` + `Cast(): incompatible function arguments` + `satmode` | [external §18](experience_classified/external_error.md) |
| `F00001` + `View(): incompatible function arguments` + `SymbolicScalar` | [function §12](experience_classified/function.md) |
| `F00001` + `transpose dims`（list 传入 transpose） | [external §12](experience_classified/external_error.md) |
| `F00002` + `Invalid data type ... for Element`（含双重包装） | [external §2](experience_classified/external_error.md) |
| `F00002` + `Not concrete value` | [external §10](experience_classified/external_error.md) |
| `F00003` + 类型注解 / `Invalid run mode` | [external §11](experience_classified/external_error.md) |
| `F0FFFF` + `AnnAssign`（类型注解） | [external §11](experience_classified/external_error.md) |
| `F0FFFF` + `Non-FP8 inputs require identical dtypes` | [matmul §2](experience_classified/matmul.md) |
| `F2FFFF` + `FE_INNER_ERROR` + `pypto.add(x, 0.0)` | [external §2](experience_classified/external_error.md) |
| `F21003` + `INVALID_TYPE: Option has invalid type` | [function §1](experience_classified/function.md) |
| `F21004` + `tile shape not set`（CAST/MUL/REGISTER_COPY） | [function §2](experience_classified/function.md) |
| `F21004` + `VEC_DUP tile shape not set`（full/zeros 前未设 tile） | [function §11](experience_classified/function.md) |
| `F21004` + `INVALID_VAL` + view 维度不匹配 | [view_op §1](experience_classified/view_op.md) |
| `F21004` + `INVALID_VAL` + `lhs.size():3, rhs.size():2` | [view_op §2](experience_classified/view_op.md) |
| `F21004` + `workspace_memory_policy does not exist` | [function §2](experience_classified/function.md) |
| `F21009` + `DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED` | [function §3](experience_classified/function.md) |
| `F21009` + `Reshape does not support dynamic axis` | [function §3](experience_classified/function.md) |
| `F21009` / `507015` + assemble 后 NaN/打乱（DYNAMIC output 问题在上游） | [function §16](experience_classified/function.md) |
| `F40005` + `TENSOR_MEMORY_ALLOCATION`（UB/L0B/L1 溢出） | [pass §1](experience_classified/pass.md) |
| `F40005` + 单 matmul 中间结果超 UB（head 分组修复） | [pass §1](experience_classified/pass.md) |
| `F40005` + FP32 替代 INT8 量化路径 → UB 放大 4× | [pass §1](experience_classified/pass.md) |
| `F40005` + Cube tile K_ext 一级过大 → L0B 溢出 | [pass §1](experience_classified/pass.md) |
| `exceeds MEM_L1` / K-tile L1 过大（K>4096 时 K_TILE 未缩小）→ L1 溢出 | [pass §1](experience_classified/pass.md) |
| `F40005` + gather 输入全量进 UB | [vector §3](experience_classified/vector.md) |
| `F4FFFF` + `Tile shape size X is not matched`（tile rank 不匹配） | [function §10](experience_classified/function.md) |
| `F4FFFF` + `REGISTER_COPY: Tile shape size X is not matched`（sub-function 泄漏 / `pypto.loop(1)` 隔离） | [function §17](experience_classified/function.md) |
| `F4FFFF` + `PASS_INNER_ERROR`（reshape inplace=False + REGISTER_COPY） | [function §2](experience_classified/function.md) |
| `F63001` + `COMPILE_CODE_FAILED`（A2A3 cast 不支持的路径） | [vector §1](experience_classified/vector.md) |
| `F7A006` + `Shape size mismatch` + `actualrawShape = [-1, ...]`（`pypto.loop` 内 `concat` + `reshape` `-1`） | [function §13](experience_classified/function.md) |
| `F7B007` + `RT_CAPTURE_FAILED 107003` | [machine §1](experience_classified/machine.md) |
| `F71008` + `MAP_REG_ADDR_FAILED`（设备资源争用） | [machine §9](experience_classified/machine.md) |
| `F76005` + `STITCH_HANDLE_INDEX_OUT_OF_RANGE` | [machine §11](experience_classified/machine.md) |
| `FC0000` + `ERR_PARAM_INVALID` + `.shape` 在 reshape 链中 | [external §8](experience_classified/external_error.md) |
| `FC0000` + `ERR_PARAM_INVALID` + kernel 参数类型注解 / `pypto.DYNAMIC` 与固定值混用 | [external §9](experience_classified/external_error.md) |
| `FC0000` + `TileShape dim num should same to input` | [function §10](experience_classified/function.md) |
| `FC0001` + `ERR_PARAM_DTYPE_UNSUPPORTED`（A2A3 cast 路径） | [vector §1](experience_classified/vector.md) |
| `FC0001` + `DIV doesn't support INT32` | [external §7](experience_classified/external_error.md) |
| `FC1001` + `ERR_CONFIG_ALIGNMENT: last axis need 32Byte align` | [vector §2](experience_classified/vector.md) |
| `FC4000` + `ERR_CONFIG_TILE: Invalid L1/L0 relation` | [matmul §1](experience_classified/matmul.md) |
| `FC4001` + `ERR_CONFIG_ALIGNMENT: must be aligned to 16 elements` | [matmul §1](experience_classified/matmul.md) |
| K 维度 cube tile kL0/kL1 设置错误（将 K/kL0 的商误当 kL1） | [pass §1](experience_classified/pass.md) |
| `F0F619` + `COMPILE_CODE_FAILED` + `make: *** Terminated`（unroll 编译超时） | [pass §2](experience_classified/pass.md) |
| `F0F619` + `Cols=1 / 32 bytes align`（RoPE nh=1 对齐） | [vector §4](experience_classified/vector.md) |
| `F0F61B` + BOOL cast 不支持 | [vector §1](experience_classified/vector.md) |
| `AC110005` / `aicore error` / `Aborted`（loop 内 SSA 重赋值） | [function §4](experience_classified/function.md) |
| `AC110005` + matmul 后 vec op tile 未重设 | [machine §3](experience_classified/machine.md) |
| `AICPU error: errorCode=0x2a, retcode: 507018`（buffer 在 loop 内创建） | [machine §2](experience_classified/machine.md) |
| `aicore error (507018 / 0x2a)` + Python `for` 展开重循环体 | [machine §4](experience_classified/machine.md) |
| `CCU instruction address check error` / `retcode 507015` | [machine §7](experience_classified/machine.md) |
| `PRECISION_FAIL` + `n_tiles boundary` | [matmul §4](experience_classified/matmul.md) |
| `PRECISION_FAIL` + L 输出精度失败，O 和 M 正常（Online Softmax） | [function §5](experience_classified/function.md) |
| `PRECISION_FAIL` + RoPE qo/ko 大面积 mismatch | [function §8](experience_classified/function.md) |
| `PRECISION_FAIL` + 大面积 OOT ~75%（view offset 超出 tile 覆盖） | [view_op §3](experience_classified/view_op.md) |
| `PRECISION_FAIL` + 跨 `pypto.loop` assemble→view 数据不可靠 | [machine §10](experience_classified/machine.md) |
| `PRECISION_FAIL` + 3 层顺序 `pypto.loop` 恰好 16 次后失败 | [codegen §1](experience_classified/codegen.md) |
| `PRECISION_FAIL` + `torch.matmul(bf16,bf16).float()` vs `pypto.matmul` | [matmul §3](experience_classified/matmul.md) |
| `PRECISION_FAIL` + `pypto.sum` 小 N 归约（tree-reduction vs sequential） | [function §15](experience_classified/function.md) |
| `SIGSEGV` / `Segmentation fault` + bare `[:]` 赋值后读取源 tensor（move 语义） | [machine §5](experience_classified/machine.md) |
| `SIGABRT` / `signal 6` + `pypto.index_add` 在 `pypto.loop` 内 | [machine §8](experience_classified/machine.md) |
| `NPU out of memory` + `stitch_function_max_num` 过大 | [pass §4](experience_classified/pass.md) |
| `exit 124 timeout` / `Pass_04→Pass_05 hang`（大权重 loop_unroll 内 cast） | [pass §5](experience_classified/pass.md) |
| `Pass_27` 后静默挂起 / `Pass_31` 超时 / `[Compiler Monitor]` >300s | [pass §2](experience_classified/pass.md) |
| `Invalid value type` + `pypto.loop()` 接收 Tensor 算术结果 | [function §9](experience_classified/function.md) |
| `K=1 matmul` cube tile 不兼容 | [matmul §5](experience_classified/matmul.md) |
| `workspace_overlap` / `non-deterministic output` | [machine §6](experience_classified/machine.md) |
| matmul 后未重置 vec tile | [external §16](experience_classified/external_error.md) |
| `CHECK FAILED: dest.GetShape().size()` / `Assemble shape mismatch` | [function §6](experience_classified/function.md) |
| `buf[a:b, :] = view(...)` 局部切片写入数据异常 / `valid_shape` 追踪失效 | [function §7](experience_classified/function.md) |
| `tensor[idx].as_variable()` 提取动态 loop 边界 | [function §14](experience_classified/function.md) |
| `could not parse dispatch key: NPU` | [external §13](experience_classified/external_error.md) |
| `from_torch()` 返回值传给 `@jit` kernel | [external §1](experience_classified/external_error.md) |
| `pypto.empty()` 不存在 | [external §1](experience_classified/external_error.md) |
| INT8 Tensor JIT 签名 format NZ vs ND | [external §15](experience_classified/external_error.md) |
| tail tile assemble 溢出（M < BM 时输出异常） | [view_op §4](experience_classified/view_op.md) |
| `PRECISION_FAIL` + attention 输出全零 + KV cache 先写后读 | [function §18](experience_classified/function.md) |
| `PRECISION_FAIL` + `index_put_` 后 `index_select` 同一 tensor | [function §18](experience_classified/function.md) |

---

## 按组件分类索引

| 组件 | 文件 | 章节数 |
|------|------|--------|
| 外部写法 | [external_error.md](experience_classified/external_error.md) | 19 |
| FUNCTION | [function.md](experience_classified/function.md) | 18 |
| MACHINE | [machine.md](experience_classified/machine.md) | 11 |
| PASS | [pass.md](experience_classified/pass.md) | 5 |
| MATMUL | [matmul.md](experience_classified/matmul.md) | 5 |
| VECTOR | [vector.md](experience_classified/vector.md) | 4 |
| 视图类OP | [view_op.md](experience_classified/view_op.md) | 4 |
| CODEGEN | [codegen.md](experience_classified/codegen.md) | 1 |
