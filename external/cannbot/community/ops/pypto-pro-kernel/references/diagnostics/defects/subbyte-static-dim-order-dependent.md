# FP4 Static Final Axis: Logical Element Contract

Fixed in pypto `172d4d915`. The defect in `50854e086`, on CANN 9.2.0 / Ascend950PR, made a fixed final axis of 64 require a Host width of 64 on a cold call while launch validation required 32; the same call therefore succeeded or failed depending on cache state.

## Current contract

- `pl.Tensor[[1, 64], pl.DT_FP4E2M1]` and FP4E1M2 both accept a `(1, 32)` byte carrier.
- `KernelSignatureSpec.bind_runtime_args` multiplies the final axis by the pack factor after reading the Host shape, then binds the shape.
- Launch validation uses the same `_pack_factor`; fixed declarations, STATIC, and ellipses all count logical elements.
- Explicit `static_shapes` are already logical shapes and are not converted for packing.
- Dynamic dimensions remain dynamic; this does not guarantee that an input meets the fixed width required by an algorithm. The caller must check the carrier.

## Regression

Framework tests `runtime/test_packed_shape_policy.py` and `runtime/jit/test_argument_conversion.py` cover binding and launch, respectively. The repository's [dtype_carriers](../../../examples/basic_api/support/dtype_carriers/test.py) test `test_static_fp4_width_is_independent_of_call_order` covers both formats, a valid cold call, a cache hit, and calls before and after an invalid call. After the fix, valid inputs must always work; the historical defect must not be turned into a rule that “static final axes are forbidden.”

Verification: 2026-09-28, CANN 9.2.0 / Ascend950PR; 26 dtype-carrier cases and 16 MX cases passed on board; Pro UT had 2557 passed and 56 skipped. Results are bound to the source and patch above; reverify after changing the toolchain.
