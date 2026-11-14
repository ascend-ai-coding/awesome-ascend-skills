# ST 用例规格

当任务包含较多用例，或契约存在不确定性，需要一份便于审查的清单时，使用 JSON 规格。简单且含义明确的小型修改可以直接写在 pytest 中。

从零交付和最终验收还应包含十个维度的覆盖评估。每个维度需记录 `covered`、`partial`、`missing`、`unknown` 或 `not_applicable` 之一，并记录支撑该结论的用例 ID 和理由。只有 `covered` 以及经过审查的 `not_applicable` 能通过完整交付门禁。

```json
{
  "coverage": {
    "functionality": {
      "status": "covered",
      "case_ids": ["tail-plus-one"],
      "rationale": "该用例比较文档要求的每一个输出值。"
    },
    "gradient": {
      "status": "not_applicable",
      "case_ids": [],
      "rationale": "公开算子没有反向计算契约。"
    },
    "execution_evidence": {
      "status": "missing",
      "case_ids": [],
      "rationale": "尚未实现该 kernel。"
    }
  }
}
```

完整键集合为 `functionality`、`precision`、`boundary`、`gradient`、`state_mutation`、`invalid_rejection`、`layout_interface`、`backend_branch`、`randomness` 和 `execution_evidence`。

## 数据结构

```json
{
  "schema_version": 1,
  "operator": "example_op",
  "contract_sources": [
    {
      "id": "api",
      "kind": "public_docstring",
      "path": "operators/example/kernel.py",
      "symbol": "example_op",
      "claim": "说明输入和输出的 shape。"
    }
  ],
  "requirements": [
    {
      "id": "R-OUTPUT",
      "statement": "每个输入元素都返回一个输出值。",
      "source_ids": ["api"],
      "applicability": "confirmed"
    }
  ],
  "cases": [
    {
      "id": "tail-plus-one",
      "kind": "boundary",
      "level": 1,
      "requirement_ids": ["R-OUTPUT"],
      "inputs": {"num_elements": 129},
      "oracles": [
        {"kind": "pytorch_reference", "source_id": "api"}
      ],
      "assertions": ["shape", "dtype", "values"],
      "path_evidence": {
        "kind": "tail",
        "axis": "num_elements",
        "logical_size": 129,
        "block_size": 128,
        "source_id": "api"
      },
      "status": "designed",
      "result": "not_run"
    }
  ]
}
```

允许的来源种类包括 `user_requirement`、`interface_doc`、`design_doc`、`public_docstring`、`api_validation`、`mathematical_definition`、`reference`、`cuda_implementation`、`ascend_implementation`、`existing_test` 和 `implementation`。

主要判定基准种类为 `pytorch_reference`、`cpu_reference`、`exact_expected`、`mathematical`、`metamorphic` 和 `exception_contract`。`cuda_differential` 与 `legacy_differential` 属于辅助判定基准，不能作为唯一的语义依据。

断言名称用于描述检查内容，并不是直接执行的代码。常用值包括 `shape`、`dtype`、`device`、`values`、`gradients`、`aux_outputs`、`state_changed`、`state_unchanged`、`protected_storage`、`exception_type` 和 `exception_message`。

生命周期状态为 `designed`、`implemented`、`collected` 和 `executed`。运行结果为 `not_run`、`passed`、`failed`、`skipped`、`xfailed` 和 `error`。只有状态为 `executed` 的用例，结果才可以不是 `not_run`；状态为 `collected` 或 `executed` 的用例必须包含 `test_nodeid`。

运行 `scripts/check_st_spec.py` 可以发现结构错误、来源引用缺失、只检查 shape 的正确性结论、只使用 CUDA 判定基准、错误的尾块标记、薄弱的负向测试以及生命周期与结果不一致等问题。该检查器通过只表示清单有效，不能证明算子正确或 pytest 已经执行。

交付门禁要求十个维度全部完成评估：

```bash
python scripts/check_st_spec.py path/to/spec.json \
  --repo-root . --require-complete-coverage
```

开启该选项后，只接受 `covered` 或经过审查的 `not_applicable`。`execution_evidence: covered` 必须引用生命周期为 `executed` 且结果为 `passed` 的用例。
