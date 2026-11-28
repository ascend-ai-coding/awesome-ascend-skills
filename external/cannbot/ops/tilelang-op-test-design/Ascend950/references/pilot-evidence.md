# 当前源码中的契约与测试示例

以下索引均相对于安装后的 TK（`repositories/Ascend950/ops-tilelang/`）。它们用于说明如何审查契约、reference 和测试，不构成当前设备上的执行通过证明。使用前核对源码版本及本地修改。

## 整数 Hash：精确比较与结构参数

- 公开接口：`src/cann_ops_tilelang/engram/engram_hash.py:engram_hash`。
- 内核：`src/cann_ops_tilelang/engram/engram_hash_kernel.py:get_engram_hash_kernel`。
- 独立 reference：`tests/engram/engram_hash_ref.py:engram_hash_ref`。
- 测试：`tests/engram/test_engram_hash.py:test_engram_hash`。

接口检查输入 Tensor 的设备、dtype、连续性及结构维度，并处理空 token 输入。测试使用 `tests/testing/numeric.py:assert_equal` 精确比较整数结果。设计用例时分别检查 token 尾块、ngram 结构、词表与 offset 的语义；不要因为测试文件存在就认为每个合法结构组合都有覆盖。

## 权重归一化：数值稳定项与多输出

- 公开接口：`src/cann_ops_tilelang/moe/normalize_weight.py:normalize_weight`。
- 内核：`src/cann_ops_tilelang/moe/normalize_weight_kernel.py:get_normalize_weight_kernel`。
- 独立 reference：`tests/moe/normalize_weight_ref.py:normalize_weight`。
- 测试：`tests/moe/test_normalize_weight.py:test_normalize_weight`。

内核输出分母和归一化权重，分母加 `1e-20`；reference 没有加这一项。现有测试使用随机非负 FP32 权重，以 `atol=2e-6, rtol=0` 比较两个输出。设计时明确零分母语义，检查空输入、top-k 维和 token 尾块，并同时验证两个输出；现有输入下的比较方式不能证明所有输入上都等价。

## TopK：索引稳定性与分支覆盖

- 公开接口：`src/cann_ops_tilelang/moe/topk_gate.py:topk_gate`。
- 内核：`src/cann_ops_tilelang/moe/topk_gate_kernel.py:get_topk_gate_kernel`。
- 独立 reference：`tests/moe/topk_ref.py:stable_topk`。
- 测试：`tests/moe/test_topk_gate.py:test_topk_gate`。

正确性测试精确比较输出索引；benchmark 单独记录时间，不能替代正确性证据。根据当前接口和内核分支确定专家数、top-k、token 尾块及重复分值用例，核对相等分值时的索引选择规则。当前测试的参数生成器使用 `tests/testing/generator.py:get_test_level`，执行证据应记录实际 `OPS_TILELANG_TEST_LEVEL` 与收集到的 nodeid。

## 交付结论

执行成功仍须满足契约、独立判定基准、有效断言和覆盖门槛。源码索引、静态发现、pytest 收集和设备运行是不同证据；缺少运行记录时保持未验证，不沿用其他项目的历史通过数或性能数据。
