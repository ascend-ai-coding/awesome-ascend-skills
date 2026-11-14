# Precision Temporary Names: Independent for Each Expansion

Fixed in pypto `8faa85389`. In the old version `50854e086`, writing the results of multiple precision operations in the same VF body to one name redefined the generated C++ temporary. This affected `ln/log/log2/log10/exp/sqrt`; ordinary mode was unaffected.

## Current contract

Callers may reuse a result variable name. `CCECodegen::GetVFTemporaryPrefix` assigns a prefix to each precision expansion; masks, temporary registers, source/destination copies, and threshold unions all use that prefix. The counter is also reset when function-generation state is reset. User variables, instruction sequence, and numerical tolerance are unchanged; precision calls no longer need distinct variable names.

## Regression and evidence

- Framework test `python/tests/ut/pypto_pro/codegen/test_precision_temporaries.py` uses the native C++ binding to check all generated CCE declarations: six operators, FP16/FP32, same name with the same op, same name with different ops, a comparison, and generator reuse.
- The repository's [vf_transcendental](../../../examples/basic_api/vf/vf_transcendental/test.py) test `test_precision_calls_can_reuse_the_destination_name` compares outputs using the same and different names and requires bitwise equality.
- On 2026-09-28 with CANN 9.2.0 / Ascend950PR, 14 duplicate-name counterexamples failed with the old library; after rebuilding and installing, 17 codegen regression cases passed and 19 transcendental example cases passed on board (one torch_npu allocation-format warning); Pro UT had 2574 passed and 56 skipped.

Precision mode and the default produce the same values in the selected positive-number domain; this does not mean the backend ignores the parameter. The backend has a subnormal-compensation path. The naming regression proves that renaming does not change results; numerical precision is still accepted against an independent CPU reference using the original tolerance.
