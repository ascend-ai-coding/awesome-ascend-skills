---
name: deepseek-fused-op-integration
description: Analyze, validate, or integrate fused training operators in DeepSeek, MindSpeed-LLM, FSDPTurbo, and other model or training repositories. Use when wiring a kernel such as TileKernels RMSNorm into a real caller, checking tensor and autograd contracts, or validating backend dispatch and correctness.
---

# DeepSeek Fused Operator Integration

Start from the requested target repository, caller and kernel. Use the target's
own dispatch, configuration and test conventions. Read its repository instructions
and inspect its revision and existing changes before editing. Use the target's
execution environment. A server or repository specified for one project does not
select the environment for another.

## Choose the work needed

- **Analysis:** inspect the actual caller, adapter, binding and relevant backward;
  report evidence and unresolved contracts. Do not imply execution occurred.
- **Validation:** use the risk-driven loop below. Do not spend a validation-only
  task comparing replacement strategies or writing a full integration proposal.
- **Implementation:** when asked to connect or replace an operator, use
  [the integration workflow](references/integration-checklist.md) to wire the
  backend through the real caller and run the smallest checks that establish reachability
  and numerical compatibility. For TileKernels RMSNorm in FSDPTurbo, start with
  [the FSDPTurbo example](references/fsdpturbo-tilekernels-rmsnorm.md).

Inspect existing changes and preserve source/checkpoints. Follow the user's
environment and download/install permissions. No system/CANN edits or distributed
training by default. Unit tests do not authorize operator source fixes.

## Risk-driven validation loop

### 1. Establish the path and contract

Locate files rather than assuming paths. Trace model caller -> imported symbol
or feature registration -> wrapper -> native forward/backward. Separate distinct
model/legacy implementations and configuration overrides.

For the chosen path record only the shapes, dtype/device, layout, defaults,
gradient requirements and parallel assumptions needed to judge compatibility.
Read branch conditions in the wrapper **and backward**, not just the signature.
Distinguish documented constraints from incidental implementation behavior.
Inspect the installed API/binding; do not invent signatures. If the real path
cannot run, report it blocked rather than replacing it and claiming kernel success.

Before generating a baseline, read the installed operator's **input constraints**
or version-matched tests, not only its signature. Write executable preconditions
for the selected fixture (including relationships between arguments). An unfused
implementation may accept inputs the native path does not. Unsupported probes
are separate from valid-input correctness tests.

Check which symbol the caller actually holds after imports/patches when a small
safe probe can establish it. If startup needs a model/distributed job outside
scope, keep reachability explicitly unverified; a standalone import is not proof
that the model uses it.

### 2. Turn source observations into falsifiable hypotheses

Before expanding a dtype/shape sweep, read
[references/risk-test-design.md](references/risk-test-design.md).
Keep a short risk ledger:

`source/branch -> expected invariant and evidence -> discriminating input -> oracle -> caller relevance`

Prioritize plausible, high-impact caller cases and branch interactions. Also test
a suspicious public/native interface combination when the source exposes a
concrete risk, but label unsupported or model-unreachable cases separately.
Do not copy every row of a generic checklist into a test matrix.

### 3. Execute an informative minimum early

After environment, input-precondition and reference checks, run one small valid
baseline through the real fused path. Only after it passes, derive the first
risk case from the same valid fixture, changing the targeted condition. For a
different native mode, establish its valid baseline first. Avoid finishing a
broad source survey or building a large custom oracle before this evidence.
Verify native dispatch with a trace, spy that still calls the original, loaded
extension evidence or equivalent; a flag alone is insufficient.

Use fresh, independent leaf tensors for each unit-test case and for the reference.
Use [scripts/tensor_fixture.py](scripts/tensor_fixture.py) or an equivalent
detach/convert/clone factory; `.to()` alone is not isolation. Check actual leaf,
requires_grad and initial grad state before backward. Keep model graph tests
separate: detaching actual model activations would change the behavior under test.

Use independent mathematical reference calculations where practical and the
existing unfused wrapper for compatibility. Agreement of the two wrappers can
miss a shared preprocessing error. Check original shape/dtype/device separately
from numerical closeness and compare relevant input/parameter gradients.
An existing CPU reference plus a small hand-checkable example is a suitable
starting point; write a new oracle when it resolves a specific shared-error risk.

### 4. Adapt to evidence, not case count

After a pass, exercise a different implementation decision or interaction before
repeating the same logic across more dtypes. Use caller-relevant precision samples
to bound numerical error; do not substitute a dtype sweep for risk coverage.
After a mismatch, minimize it, verify the oracle/support assumptions, and trace
the condition back to a caller. First check fixture isolation and input validity;
do not investigate downstream mismatches caused by an invalid upstream result.
Preserve the original output. Do not loosen
tolerances or change expected behavior merely to pass.

Respect the agreed budget. Reserve time for running and interpreting tests;
freeze exploration when the budget is nearly spent. If setup/JIT blocks execution,
report the blocker and highest remaining risk instead of filling out a proposal.
If a fixture fails, pause the sweep and repair only the baseline within budget.
Keep invalid evidence separate from corrected results; a passing different mode
does not validate the original mode. Record repair time and incomplete coverage.

## Evidence and decisions

Use [references/test-checklist.md](references/test-checklist.md) for reference,
failure-status and execution requirements. Use the existing
[scripts/run_validation.py](scripts/run_validation.py) across shells to record
the child status; it cannot detect a harness that swallows assertion failures.

Classify results as: numerical failure, interface/state discrepancy, supported
behavior, unsupported combination, unresolved contract, blocked, or not tested.
An unexpected assertion/exception fails its check; an unresolved oracle must not
be relabelled PASS or presented as a proven kernel/model defect.

Deliver findings with the trigger, expected/actual behavior, evidence for the
expectation, actual path tested and caller relevance. Attach commands, environment,
references/tolerances, raw output and child exit status. State uncovered high-risk
branches. Do not require a nine-section report for a small validation task.
Whole-model, distributed, performance and checkpoint claims require their own
authorized execution. No executed issue is also a legitimate outcome; do not
invent a defect to justify this skill.
