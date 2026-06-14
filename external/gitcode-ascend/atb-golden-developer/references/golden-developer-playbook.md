# Golden 参考实现 Playbook

本文为 `atb-golden-developer` 延伸阅读：NO_ERROR 流程、三件套、hostData、调试与示例代码。

## NO_ERROR vs I:NO_ERROR

CSV 测试框架通过 `ExpectedError` 列的阶段前缀控制测试流程：

| ExpectedError | C (Create) | I (InferShape) | S (Setup) | E (Execute) | G (Golden) |
|--------------|------------|----------------|-----------|-------------|------------|
| NO_ERROR | 运行 | 运行 | 运行 | 运行 | 运行 |
| I:NO_ERROR | 运行 | **停止** | - | - | - |
| C:ERROR_xxx | **停止** | - | - | - | - |

**关键结论**:
- **I:NO_ERROR 不执行 kernel、不运行 golden** — 仅验证 shape 推导
- **NO_ERROR 需要完整的 C→I→S→E→G 流程** — kernel 执行 + golden 比较
- NO_ERROR 的数据生成**必须用 `customize` 模式**，不能用 `random`
  - 原因：Setup 阶段需要通过 `hostData` 注入标量参数（contextLens、qSeqlen）
  - `random` 模式生成的 NPU tensor 没有 hostData → `BuildFromTensor` 失败 → `S:ERROR_INVALID_PARAM`
  - `customize` + `case_preprocess` 通过 hosttensor binder 注入 hostData

### 数据生成模式对比

| 模式 | CSV data_gen 列 | hostData 注入 | 适用 |
|------|----------------|-------------|------|
| `random` | `random;random;...` | 无 | I:NO_ERROR、反例 |
| `customize` | `customize;customize;...` | 需要 `case_preprocess` | NO_ERROR |

---

## Kernel 对齐原则

**Golden 必须验证 kernel 的实际行为，不能基于假设编写。**

### 为什么需要对齐

Golden 的目标是精确复现 kernel 的计算结果。如果 golden 假设了 kernel 不执行的操作，输出会完全不同。

### 通用对齐步骤

1. **定位 kernel 源码**：在 ATB 仓库中找到目标算子的 AscendC 实现（如 `.cce`），核对分支、mask、dtype 转换等关键路径。
2. **对照参考实现**：优先使用同仓库内的 kernel 测试或 apitest 中的 `ref_*` / `golden_calc` 等已验证逻辑，与 CSV golden 逐项对比。
3. **核对参数语义**：mask type、tiling、decode/prefill 等是否存在多条路径；不要假设枚举值均有完整实现。
4. **按算子沉淀案例**：具体源码路径、Issue 背景与踩坑记录放在 `references/` 下的案例文档（见下方），技能正文不承载单算子硬编码路径。

### 案例索引（按算子）

- MLA / PagedAttention（Issue #288，SWA mask 与 kernel 分支）：  
  [references/case-study-mla-paged-attention-kernel-alignment.md](references/case-study-mla-paged-attention-kernel-alignment.md)

### 验证清单

- [ ] 已阅读 kernel 源码的关键分支（mask 应用、dtype 转换、softmax 实现）
- [ ] 已对比 kernel 参考测试的 golden 逻辑
- [ ] 已在不同数据范围（-0.1~0.1 和 -5~5）测试通过
- [ ] 已确认 mask type 在 kernel 中的处理路径

---

## Golden 开发三件套

一个完整的 DataGen 类必须实现以下 3 个方法：

```
DataGen 类 (data_generation.py)
├── customize()      → 生成输入 tensor，返回 NPU tensor
├── golden()         → CPU 参考计算，返回 [cpu_tensor]
└── case_preprocess() → 注入 hostData，让 runner 能读取标量参数
```

### customize — 生成测试数据

```python
@staticmethod
def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
    # shapes: 所有输入 tensor 的 shape 列表
    # i: 当前生成的 tensor 索引 (0 ~ InNum-1)
    # datatype: CSV 中指定的 dtype (如 "float16", "int32")
    # 返回: 单个 NPU tensor

    if i != 0:
        # 非首次调用: 返回预存的 tensor i（索引越界须立即失败，禁止静默回退到 i==0 分支）
        if i >= len(cls.in_tensors):
            raise ValueError(
                "customize: i=%d but len(in_tensors)=%d；检查 CSV InNum 与首次 customize(i==0) 是否已执行"
                % (i, len(cls.in_tensors))
            )
        return cls.in_tensors[i]

    # i == 0: 生成全部 tensor，存入类变量
    # 关键: 必须用 numpy → torch.from_numpy → .npu() 路径
    query_np = np.random.uniform(min_value, max_value, size=(...)).astype(np.float32)
    bt_np = np.zeros((...), dtype=np.int32)
    ctx_np = np.full((...), 128, dtype=np.int32)

    cls.in_tensors = [
        torch.from_numpy(query_np).to(torch.float16).npu(),  # NPU tensor
        ...
        torch.from_numpy(bt_np).npu(),                        # int32 NPU tensor
        torch.from_numpy(ctx_np).npu(),
    ]
    cls.golden_tensors = [
        torch.from_numpy(query_np),   # CPU float32 tensor
        torch.from_numpy(bt_np),      # CPU int32 tensor
        ...
    ]
    return cls.in_tensors[0]
```

**规则**:
- `cls.in_tensors`: NPU tensor 列表（给 ATB 算子执行用）
- `cls.golden_tensors`: CPU tensor 列表（给 golden 参考计算用）
- 必须用 `numpy → torch.from_numpy → .npu()` 创建 NPU tensor
- i==0 生成全部，i>0 返回预存
- 若 `i>0` 且对应 `in_tensors[i].numel()==0`，须**显式** `raise` 或返回合法 tensor，**禁止**落到函数末尾隐式返回 `None`

### golden — CPU 参考计算

```python
@staticmethod
def golden(in_tensors, op_params):
    # in_tensors: 框架传入的 CPU tensor 列表（自动 .cpu()）
    # 但推荐从 cls.golden_tensors 读取（与 customize 生成的数据一致）
    # 返回: [torch.Tensor.cpu()]

    gt = cls.golden_tensors
    query = gt[0].float()
    ...

    # 计算参考输出
    output = compute_attention(query, key, value, mask)

    # 清理类变量
    del cls.in_tensors
    del cls.golden_tensors
    return [output.cpu()]
```

**规则**:
- 输入是 CPU tensor（框架自动 `.cpu()` 转换）
- 输出必须是 CPU tensor（`.cpu()`）
- 计算完成后清理类变量

### case_preprocess — 注入 hostData

> **算子特定槽位**：下面 `input_tensor_list[5]`、`input_tensor_list[7]` 仅对应 **PagedAttention / MLA 等** 在该测试 harness 中的输入排布。其它算子必须对照本算子 `InferShape` 的 **tensor 顺序** 自行选择下标；直接复制会导致**静默写错 host 数据**或 `IndexError`。实现前请打开目标算子的 hosttensor binder（`ParseParam` 期望哪些键）并对齐槽位。

```python
@staticmethod
def case_preprocess(op_params, operation, input_tensor_list):
    # 在 InferShape 之后、Setup 之前调用
    # 用途: 将标量数据 (contextLens, qSeqlen 等) 注入 variant pack
    # 这解决了 Setup 阶段 "hostData is null" 的根本问题

    json_data = json.loads(op_params)
    # calcType 来自算子 JSON；若字段缺失则按 0 处理（请与你的算子 schema 对齐）
    calc_type = int(json_data.get("calcType", 0))
    host_dict = {}
    # --- 以下为 PagedAttention/MLA 类示例槽位；替换为你的算子实际索引 ---
    _idx_ctx, _idx_qseq = 5, 7
    _need = _idx_ctx + 1
    if calc_type in [1, 3]:
        _need = max(_need, _idx_qseq + 1)
    if len(input_tensor_list) < _need:
        raise IndexError(
            f"input_tensor_list 长度 {len(input_tensor_list)} < {_need}，请核对槽位映射（示例 ctx={_idx_ctx}, qseq={_idx_qseq}）"
        )
    host_dict["contextLens"] = input_tensor_list[_idx_ctx].tolist()
    if calc_type in [1, 3]:  # SPEC or SPEC_AND_RING
        host_dict["qSeqlen"] = input_tensor_list[_idx_qseq].tolist()
    if "maskType" in json_data:
        host_dict["maskType"] = json_data["maskType"]

    run_param = json.dumps(host_dict)
    operation.set_varaintpack_param(run_param)
```

**机制**: `set_varaintpack_param` 触发 hosttensor binder（与 ATB 测试框架 Python 绑定名一致，见仓库内 `tests/framework/c++/atb_torch/operation/operation_torch.cpp` 中对 `set_varaintpack_param` 的导出；若未来 C++ 侧改名，以你检出版本的 `Operation`/`OperationTorch` 为准）:
1. 框架查找 `hosttensor_binder_creator.cpp` 中注册的 binder
2. binder 的 `ParseParam(json)` 解析 JSON
3. 在 Setup 时，binder 的 `BindTensor(variantPack)` 将数据写入 `tensor.hostData`
4. runner 的 `BuildFromTensor` 可以成功读取 hostData

---

## hostData 注入机制

### 原理

ATB runner 在 Setup 阶段需要从 host 侧读取标量参数（如 contextLens、qSeqlen）。NPU tensor 默认不携带 hostData，需要通过 **hosttensor binder** 显式注入。

### binder 文件位置

```
tests/framework/c++/atb_torch/operation/
├── hosttensor_binder_creator.cpp    ← 注册 binder 的工厂函数
└── hosttensor_binders/
    ├── multi_latent_attention_binder.cpp  ← 示例: MLA 的 binder
    ├── pagedattention_binder.cpp
    └── ...
```

### 如何确定 JSON 格式

阅读对应算子的 binder `ParseParam` 方法。以下是**示意结构**（请将 `XxxBinder` 替换为目标算子在 `hosttensor_binders/` 下的实现）：
```cpp
void ParseParam(const nlohmann::json &paramJson) {
    if (paramJson.contains("contextLens")) { ... }
    if (paramJson.contains("qSeqlen")) { ... }
    if (paramJson.contains("maskType")) { isMask_ = true; }
    if (paramJson.contains("cacheType") && paramJson["cacheType"] == 1) { isInt8Nz_ = true; }
}
```

### binder 未注册时

如果 `hosttensor_binder_creator.cpp` 中没有对应算子的 binder，`set_varaintpack_param` 无效，需直接修改 runner 代码或添加新 binder。

---

## 调试流程

### 错误分层定位

测试框架按阶段拦截错误（ExpectedError 前缀）:

| 阶段 | 前缀 | 常见错误 |
|------|------|---------|
| Create | `C:` | ERROR_INVALID_PARAM, ERROR_INVALID_TENSOR_DIM |
| InferShape | `I:` | ERROR_INVALID_TENSOR_DIM, ERROR_INVALID_TENSOR_DIM_NUM |
| Setup | `S:` | ERROR_INVALID_PARAM (hostData null) |
| Execute | `E:` | ERROR_RT_FAIL |
| Golden Compare | (无前缀) | 精度不达标 |

### 开启详细日志

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0        # DEBUG
export ASCEND_MODULE_LOG_LEVEL=OP=0     # ATB 模块 DEBUG
export ASCEND_PROCESS_LOG_PATH=/path/to/log
```

查看日志:
```bash
grep -i "error\|hostData\|contextLens\|build param" /path/to/log/atb/*.log
```

### 常见错误速查

| 错误信息 | 根因 | 修复 |
|---------|------|------|
| `hostData is null` | tensor 缺少 host 侧数据 | 添加 case_preprocess |
| `build param from host tensor fail` | BuildFromTensor 读到空 hostData | 同上 |
| `size a (N) must match size b (M) at dim K` | matmul 维度计算错误 | 检查 permute/transpose |
| `I:ERROR_INVALID_TENSOR_DIM` | tensor shape 不满足 DimCheck | 对照 operation.cpp 的 DimCheck 逐项核对 |
| `I:ERROR_INVALID_IN_TENSOR_NUM` | InNum 不对 | 对照 op 的 GetInputNum() |

### Matmul 维度陷阱

多 head attention 的 golden 计算中，常见错误是将 `[ctx, nh, d]` 直接 transpose 为 `[ctx, d, nh]` 后与 `[1, nh, d]` 做 matmul，导致 batch 维度广播出 `[ctx, nh, nh]` 而非预期的 `[1, nh, ctx]`。

**正确做法**：使用 `permute` 将 K 转为 `[nh, d, ctx]`，然后用 per-head batched matmul:
```python
kt = k_full.permute(1, 2, 0)   # [nh, d, ctx]
scores = matmul(q.unsqueeze(1), kt)  # q:[nh,1,d] × kt:[nh,d,ctx] → [nh,1,ctx]
```

---

## get_op_type 与精度标准

### get_op_type 方法

```python
@staticmethod
def get_op_type(op_params) -> OpTypes:
    return OpTypes.CV_FUSION  # 根据算子类型选择
```

### 常用 OpTypes

| OpType | 精度标准 | 适用场景 |
|--------|---------|---------|
| COMPUTE_FLOAT | 逐元素阈值 (2^(-8) for fp16) | 简单计算 |
| CV_FUSION | MARE/MERE/RMSE + Error1‰ | Attention 类融合算子 |
| VECTOR_FUSION | 逐元素阈值 (2^(-8) for fp16) | CV_FUSION 降级后备 |
| COMPUTE_FLOAT_HIGH_PRECISION | 高精度阈值 | 高精度计算 |

### CV_FUSION 自动降级

当使用 `--precision_standard new` 且未提供 GPU 信息时，框架自动将 CV_FUSION 降级为 VECTOR_FUSION：

测试框架在「`--precision_standard new` 且未提供 GPU 信息」等条件下，可能将 `CV_FUSION` 自动降级为 `VECTOR_FUSION`。逻辑位于 `atb_csv_ops_test.py`（请在本机 ATB 仓库内用 `rg "precision_standard|CV_FUSION|VECTOR_FUSION" tests/framework/python/CsvOpsTestTool` 定位；**勿依赖文档中的固定行号**）。

**建议**：开发阶段使用 `--precision_standard old`（Error1‰ 标准），直观且无需 GPU 信息。

---

## 精度调试

### 两个精度标准

| 标志 | 标准 | 输出项 | 阈值 (fp16) |
|------|------|--------|-----------|
| `--precision_standard old` | Error1‰/Error4‰/Error5‰ | Error0.1‰~Error+/-1 | Error1‰≥99.9% |
| `--precision_standard new` | MARE/MERE/RMSE/EB | PrecisionPercent, EBPercent | PrecisionPercent≥100% |

### 查看精度数值

运行测试后，读取结果 CSV 的精度列：

```bash
grep "^<CaseNum>" multi_latent_attention_csvopstest_result.csv | tr '|' '\n' | grep "%"
```

输出示例（old 标准）：
```
Error0.1‰: 46.75%
Error0.5‰: 99.43%
Error1‰: 100.0%     ← 判定标准
Error4‰: 100.0%
Error5‰: 100.0%
Error+/-1: 100.0%
```

### 精度不达标排查

1. **对比仓库内参考实现**：在 ATB 仓库内搜索 kernel/apitest 中已有的 `ref_*`、`golden_calc` 或与当前算子同系列的测试（例如 `rg "ref_" tests/apitest`）。涉及 MLA/PagedAttention、SWA mask 与 kernel 分支对齐的**完整案例**见 [references/case-study-mla-paged-attention-kernel-alignment.md](references/case-study-mla-paged-attention-kernel-alignment.md)，勿把单一测试文件当作所有算子的必经步骤。
2. **检查 dtype 转换**: golden 输入是否经过 float16 截断 (`.half().float()`)
3. **检查 scale**: qkScale 是否与 kernel 的 `tor` 一致
4. **检查 mask**: kernel 是否实际应用了 golden 假设的 mask
5. **检查 matmul 维度**: 确认 permute/transpose 方向正确
6. **检查 MQA broadcast**: repeat_interleave vs group_matmul

### 日志级别

```bash
# INFO 级别（查看精度数值）
python3 atb_csv_ops_test.py -i <CSV> -n <case> --log_level info --precision_standard old

# DEBUG 级别（查看 mare/mere/rmse/EB）
python3 atb_csv_ops_test.py -i <CSV> -n <case> --log_level debug --precision_standard new 2>&1 | grep "mare\|mere\|rmse\|EB"
```

---

## 完整示例

在 ATB 仓库的 `data_generation.py`（路径随版本可能调整）中，用类名检索具体实现，**不要使用固定行号**（行号会随合并漂移）：

```bash
cd <ATB_REPO_PATH>
rg -n "^class YourOperation" tests/apitest/opstest/python/data_generation.py
```
（将 `YourOperation` 替换为目标 DataGen 类名；若仓库中 `data_generation.py` 路径不同，先用 `find` / `rg --files -g data_generation.py` 定位。）

复杂算子（含 hostData、decode/prefill、mask 多分支）的**仓库内定位与历史案例**见 [references/case-study-mla-paged-attention-kernel-alignment.md](references/case-study-mla-paged-attention-kernel-alignment.md) 中的「DataGen 参考类」小节。

### 最小实现模板

```python
class YourOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            if i >= len(YourOperation.in_tensors):
                raise ValueError(
                    "customize: i=%d but len(in_tensors)=%d"
                    % (i, len(YourOperation.in_tensors))
                )
            t = YourOperation.in_tensors[i]
            if t.numel() == 0:
                raise ValueError(
                    "customize: input index %d is empty tensor; fix CSV 或前序 tensor 生成逻辑"
                    % i
                )
            # 将 CSV `format` 枚举映射到 CANN `npu_format`（占位示例，按算子与头文件自行填写）
            format_to_acl = {0: 2, 1: 29}
            acl_fmt = format_to_acl.get(int(format), 2)
            return torch_npu.npu_format_cast(t, acl_fmt)

        # i==0: 生成全部 tensor
        query_np = np.random.uniform(-0.02, 0.02, size=shapes[0]).astype(np.float32)
        ...
        YourOperation.in_tensors = [torch.from_numpy(query_np).to(torch.float16).npu(), ...]
        YourOperation.golden_tensors = [torch.from_numpy(query_np), ...]
        return YourOperation.in_tensors[0]

    @staticmethod
    def golden(in_tensors, op_params):
        gt = YourOperation.golden_tensors
        # CPU 参考计算...
        output = compute_reference(gt)
        del YourOperation.in_tensors
        del YourOperation.golden_tensors
        return [output.cpu()]

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        # 与上文「case_preprocess — 注入 hostData」一致：按 binder `ParseParam` 构造键；
        # 不得照抄示例下标，须按本算子 input_tensor_list 顺序填写。
        json_data = json.loads(op_params)
        host_dict = {}
        # 例: host_dict["contextLens"] = input_tensor_list[<你的槽位>].tolist()
        operation.set_varaintpack_param(json.dumps(host_dict))

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT
```

