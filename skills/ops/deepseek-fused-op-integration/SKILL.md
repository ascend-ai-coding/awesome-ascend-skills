---
name: deepseek-fused-op-integration
description: Analyze and integrate fused operators into DeepSeek models in MindSpeed-LLM. Use when locating operator insertion points, analyzing tensor contracts, choosing monkeypatch or direct replacement, implementing Ascend NPU fused operator integration, or validating operator correctness.
---

# DeepSeek Fused Operator Integration

## Objective

Provide a reusable workflow for integrating fused operators
into DeepSeek model implementations in MindSpeed-LLM.

The workflow must not be limited to one specific operator.

Reference repository:
https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/mindspeed_llm/core

Always inspect the checked-out local source code before
making implementation decisions.

## Workflow

### Phase 1: Locate the Algorithm

1. Identify the model architecture and target operator.
2. Search for the original algorithm implementation.
3. Trace the call chain from model entry to operator call.
4. Identify the relevant layer, module, and function.
5. Determine the position of the fused operator.

Record:

- Model name
- Source file
- Module/class/function
- Original computation
- Proposed insertion point
- Forward and backward dependencies

Do not assume a source path exists without checking.

### Phase 2: Analyze the Tensor Contract

Identify the inputs and outputs of the original computation.

For every tensor, record:

- Name
- Shape
- Dtype
- Device
- Layout
- Contiguity constraints
- Semantic meaning
- Gradient requirements

Analyze shape transformations before and after fusion.

Pay attention to:

- Batch dimension
- Sequence dimension
- Hidden dimension
- Attention head dimension
- Tensor parallel partitioning
- Reshape and transpose operations
- Optional arguments and dynamic shapes

Identify whether the fused operator interface exactly
matches the original computation.

If not, define the required adapter.

### Phase 3: Analyze the Fused Operator

Locate the implementation or API of the fused operator.

Inspect:

1. Python interface
2. C++ extension or operator binding
3. Ascend CANN/aclnn interface
4. Supported input shapes and dtypes
5. Output definitions
6. Backward implementation
7. Device and version requirements
8. Known operator limitations

Do not invent API signatures.

If an implementation cannot be located, report the
missing information before proceeding.

### Phase 4: Select the Integration Strategy

Evaluate two primary approaches.

#### Strategy A: Monkeypatch

Use when:

- The original call can be safely replaced at runtime.
- The target implementation should remain unchanged.
- The integration is optional.
- The function or module can be patched reliably.

Check:

- When the original symbol is imported
- Whether other modules retain references to it
- When the patch is applied
- Whether the patch affects unintended callers
- How to restore the original behavior

#### Strategy B: Direct Replacement

Use when:

- The framework implementation is under control.
- The fused operator requires structural modifications.
- Runtime patching is unreliable.
- Explicit model integration is preferred.

Check:

- Compatibility with existing configurations
- Original implementation fallback
- Impact on model initialization
- State dict and checkpoint compatibility
- Training and inference behavior

#### Decision Output

Generate a comparison including:

- Chosen strategy
- Rationale
- Files to change
- Compatibility risks
- Rollback mechanism

Never choose monkeypatch or direct replacement
without inspecting the actual call chain.

### Phase 5: Implement the Integration

Follow the existing project conventions.

1. Create an adapter if necessary.
2. Preserve the original public interface where practical.
3. Introduce an explicit feature flag.
4. Add shape and dtype checks.
5. Preserve autograd behavior.
6. Retain a safe fallback when supported.
7. Integrate the fused operator at the chosen location.

Do not modify repository files until the user explicitly
requests implementation.

Do not silently change model semantics.

### Phase 6: Validate the Operator

Validation must include:

1. Import and availability checks
2. Forward output shape checks
3. Forward dtype checks
4. Numerical correctness against a reference
5. Backward gradient correctness where applicable
6. Representative boundary shapes
7. Training and inference compatibility
8. Distributed parallel compatibility when relevant
9. Performance measurement when a baseline is available

Use appropriate dtype-specific numerical tolerances.

Do not report a test as passed unless it was executed
and its output was inspected.

### Phase 7: Report Results

Produce an integration report containing:

1. Algorithm and operator overview
2. Source code location
3. Original computation and fused computation
4. Tensor input/output table
5. Operator call chain
6. Integration strategy comparison
7. Implementation proposal or changes
8. Test results
9. Risks, limitations and rollback plan

Separate verified facts from assumptions.

## Safety Rules

- Start with read-only source analysis.
- Never modify system or CANN installation files by default.
- Never overwrite model checkpoints.
- Never initiate distributed training automatically.
- Request explicit approval before editing code.
- Avoid destructive shell commands.
- Prefer independent unit tests before model-level tests.

## Additional References

Read references/integration-checklist.md when planning integration.

Read references/test-checklist.md when preparing validation.

## Example Requests

- Analyze where a fused attention operator should be inserted.
- Determine the input/output contract of an mHC operator.
- Compare monkeypatch and direct replacement for a DeepSeek module.
- Generate an integration plan for an Ascend NPU fused operator.
- Validate an integrated operator against the original computation.
