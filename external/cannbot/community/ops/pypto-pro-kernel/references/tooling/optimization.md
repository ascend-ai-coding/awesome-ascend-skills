# Optimization candidates and current experiment state

Use these commands after identifying a correct baseline. [The optimization workflow](../workflows/optimization.md) owns experiment selection; this page owns CLI inputs, evidence gates and state. All paths below are explicit caller paths. Write state, specs and observations in ignored scratch or outside the repository. Nothing is created by `suggest`; `experiment` writes only its requested state and companion lock, and exports source only on request.

## Bounded candidate lookup

```bash
python /path/to/skill/scripts/check.py suggest --source candidate.py
python /path/to/skill/scripts/check.py suggest --source candidate.py --source helper.py --measurements /scratch/costs.json --limit 3
```

The JSON feedback returns at most three candidates by default (`--limit` accepts 1–5), counts omitted sites and names unresolved scope. Each candidate includes its source hash/location, applicability conditions, cost terms expected to decrease/increase, rejection conditions, a discriminating comparison and the owner query. The prediction's `latency_delta_us` remains null: source syntax does not establish elapsed-time savings. Follow `query` to read that owner's mechanism and complete example; the suggestion does not embed whole reference chapters.

| Covered pattern | Evidence and boundary |
| --- | --- |
| `vf-loop-runtime-control` | Reuses the [VF loop advisory](checks.md#static-rules). Call-site constants, recurrence meaning and native loop lowering remain unproved. |
| `nested-worker-restart` | Reuses the [grouped ownership advisory](checks.md#static-rules). Actual group lengths and per-worker work must establish an imbalance before comparing ownership. |
| `ub-l1-extra-layout-move` | Uses direct source descriptors to find an ND→NZ Vec move followed by a same-expression Vec→Mat handover in one JIT. This lexical chain does not prove the active branch, physical identity, producer bytes or other consumers. |

Recipe fields live in the `candidate` object of the owning section's `kernel-query` marker, alongside the full explanation. Do not maintain a parallel candidate catalog or infer a rule from arbitrary prose. Missing matches mean only that these three families did not match; they are not evidence of a hardware ceiling. The command proposes comparisons, does not edit kernels and does not replace `rules`, `native` or numerical checks.

Use the [structural candidate-generation procedure](../workflows/optimization.md#generate-a-concrete-alternative-from-the-computation) independently of these matches. The tool does not infer algebraic simplifications, cross-phase residence, alternative output ownership or the best publication boundary from arbitrary source. A returned local instruction candidate need not have higher priority than a source-grounded structural rewrite. `select_experiment` guidance asks for that review while preserving the caller's fixed-family scope; it does not certify or automatically execute the rewrite.

Identify the [baseline family's controls](../workflows/optimization.md#distinguish-family-selection-from-an-implementation-change) and [expand the actual expressions](../workflows/optimization.md#derive-opportunities-from-actual-expressions) before ranking a local candidate. Existing-parameter selection can change realized ownership without implementing a new dataflow. The source detector does not perform that classification or prove a contraction invariant; the workload and complete producer expression remain necessary evidence.

Optional `pypto-kernel-costs/1` input:

```json
{
  "schema": "pypto-kernel-costs/1",
  "sources": [{"path": "/scratch/candidate.py", "sha256": "<actual source SHA256>"}],
  "framework_sources": {"<selected framework relative path>": "<actual file SHA256>"},
  "toolchain": {"path": "/scratch/toolchain.json", "sha256": "<actual identity-file SHA256>"},
  "scope": "complete_call",
  "workload": {"shape": [64, 128], "workers": 1, "input_regime": "coordinate-distinct, resident inputs"},
  "profile": {},
  "sites": [{"file": "/scratch/candidate.py", "line": 71, "task_position": 1}]
}
```

Replace `profile` with the unchanged result of [`measure_pipes` or `measure_native`](../verification/native-timing.md); an empty object is invalid. `framework_sources` is `feedback.identity(pypto_root)["selected_source"]["files"]` from the selected installation's matching source tree. `sources` must exactly match the requested source set, including helpers; paths in this file resolve relative to its directory. The tool verifies source bytes, framework fingerprints and the toolchain identity file. The identity file should include actual loaded framework/native-library hashes, compiler/SDK identity and device target, not just a package version.

`sites` explicitly maps source lines to native task positions; this mapping comes from the execution driver/generated program, not from the AST. Matched candidates sort by measured task median, with unmapped sites afterward. Task duration is not a site's exclusive cost. Complete-call medians, task medians and pipe ratios have different meanings; they are not added or converted into predicted savings. Missing counters remain null. Stale or invalid costs produce `unverified`/exit 2 and leave candidates unranked by measurements; they never silently fall back to historical costs.

## Optional performance-card source

Pass `--cards-root references/knowledge-cards` before the command to search the [bundled Active cards](../knowledge-cards/index.md) from the skill directory (use an absolute path from elsewhere). Install `scripts/requirements-cards.txt` into the checker interpreter only when this optional source is used. The CLI keeps this source opt-in; ordinary document search does not admit unregistered cards or contributor examples.

```bash
python scripts/check.py --cards-root references/knowledge-cards query cube-06 --kind document --format json
python scripts/check.py --cards-root references/knowledge-cards query 'document:perf-cards/cube/cube-06-grouped-continuous-core-assign.md' --format json
python scripts/check.py --cards-root references/knowledge-cards query --validate --format json
python scripts/check.py --cards-root references/knowledge-cards suggest --source candidate.py
```

Only files registered as stable in the Active table are collected. The provider returns `document_role=performance_card`, the original card and index identities, `card.item_id`, original applicability/API gates, exact document reads and declared relationships. `knowledge_links` uses `pypto-knowledge-links/1` with `same_mechanism`, `overlaps_with`, `uses_contract` or `alternative_to`, a target skill/object ID, and explicit `scope`/`boundary`. Relationships remain local to their declared scope and are not transitive. They do not copy applicability facets or acceptance evidence. Unconfigured cards leave the base skill independent; an explicitly configured unavailable/invalid library reports `unverified` with coverage errors and quarantines invalid cards. Full admission still requires the matching framework source.

`coverage.documents.cards` reports `active_items`, `cards_read`, `linked_cards`, `unlinked_cards` and `relation_count`. The last three count only accepted cards and resolved relationships; an invalid card still contributes to the Active count and explicit errors. A readable card without relationships remains queryable. Complete reads and resolved links establish retrieval coverage, not implemented detectors or verified optimizations. Read the card's full conditions and capability gaps before choosing an experiment; a document relationship can cover only one prerequisite of a broader technique.

For partial card content, pass the returned SHA-pinned `next` request to `read --request` with the same `--cards-root` configuration. This explicitly selected root is available to default context reads even outside the execution workspace. An explicit context policy's `roots` remains authoritative; include the card directory there when that policy should permit continuations.

`suggest` attaches directly linked cards to existing source-detected candidates as `related_cards`. `same_mechanism_only` groups references under that candidate without creating a duplicate, while specific candidate equivalence and current applicability remain `not_established`; other relationships are `reference_only`. A contract reference does not make every optimization inside the contract the same intervention. There are still only three source-pattern families: other cards need a query and explicit applicability judgment, not an invented source match or predicted speedup. Original card Markdown reads use `document:perf-cards/<registered-relative-path>`, with the same SHA-pinned read and source-sensitive cursor rules as other documents.

## Action-scoped guidance

An explicit `--goal` on `query` or `suggest` adds `action_guidance` without execution, a new state file or a knowledge cache. Goals are `select_experiment`, `native_probe`, `validate_numerics` and `compare_performance`. A controller task inventory may declare the same optional `goal`; a conflicting explicit goal is invalid. The inventory still verifies only declared bytes and labels inline evidence as unauthenticated.

```bash
python scripts/check.py query acc-final-output --source candidate.py --goal select_experiment --format json
python scripts/check.py suggest --source candidate.py --goal select_experiment
```

Guidance separates `required_now` from `later_checks`. A resolved scenario can support a conditional hypothesis while installation, numerical and timing evidence remain outstanding. Check applicability against the actual workload and state a rejection test. Once those facts suffice, run or describe the next discriminating probe; reading more declarations cannot establish execution or timing. An exact declaration alone does not establish a producer/consumer route. Missing contracts, unresolved selected tables, changed inventories, source errors and stale measured rankings remain explicit.

`readiness` describes only the requested action; it is not permission, kernel acceptance, proof of a speedup or proof that optimization should stop. Original `validation` and `missing_facts` retain their evidence boundaries. No previously delivered result is assumed to survive a new conversation or context compaction; full source/reference content remains available. Query and suggestion behavior is unchanged when no goal is supplied.

During an experiment, the existing best/current/phase state is authoritative. Its read-only summary derives `action_guidance` from that phase: complete missing baseline/candidate validation, collect required timing, select the next hypothesis from best, or export best with open directions when selections are exhausted. A numerically rejected candidate first calls for restoring the verified best and inspecting the failed original-evaluator cases, including at zero remaining selections; a timing rejection does not imply numerical failure. Pending evidence can still be completed at zero remaining selections. The projection is not persisted and creates no parallel trial history.

## Performance claim consistency

`claim --spec /scratch/claim.json` checks a positive assertion or a supplied claim under review. It reads declared snapshots and an explicit observation; it does not run kernels, update experiment state, authenticate a verifier or prove hardware performance. A decision making no performance claim does not require another query/check loop. The checker reuses the experiment's identity, numerical coverage and AB/BA retention gates.

```json
{
  "schema":"pypto-performance-claim/1",
  "claim":{"speedup_established":true,"workload":{"rows":8,"cols":256},"metric":"synchronized_wall_us","changes_wrapper":false},
  "baseline":{"root":"baseline","files":["kernel.py"],"entry":"kernel.py:kernel","configuration":{}},
  "candidate":{"root":"candidate","files":["kernel.py"],"entry":"kernel.py:kernel","configuration":{}},
  "toolchain":{"path":"toolchain.json","sha256":"EXPECTED_TOOLCHAIN_SHA256"},
  "observation":{"path":"observation.json","sha256":"EXPECTED_OBSERVATION_SHA256"}
}
```

The baseline/candidate declarations use the snapshot form below; hashes include every declared filename, entry, configuration and source byte. Paths resolve beside the claim specification. The observation uses `pypto-experiment-observation/1` below and must additionally declare `evidence_kind: native_measurement` for a native assertion. This is a verifier attestation label, not authentication; a synthetic fixture or user statement cannot be promoted by the checker. The target workload must match a concrete observed and timed case exactly, including JSON value types; a workload label alone is insufficient. The metric, actual variant hashes, toolchain, original numerical oracle/coverage and paired complete-call evidence must match. Wrapper/task-count changes keep the existing wall-time requirement. Omitted evidence remains unknown.

The JSON result's `claim_check` is `supported`, `unsupported`, `contradicted` or `not_claimed`. `supported_speedup` is true only when the supplied matching attestation passes retention, false when its numerical/timing gates reject the candidate, and null for missing, mismatched, partial or inconclusive evidence. A false input flag produces `not_claimed`; it does not prove that improvement is impossible or that optimization should stop. CLI exit 0 means supported/not-claimed report handling, exit 1 means a contradicted positive assertion, and exit 2 means unsupported/invalid input. `validation` retains `not_independently_established`: even a supported consistency check does not independently replay or authenticate the observation. Preserve that distinction in the final report and do not rerun work merely because its evidence was unavailable.

## One best snapshot and one current trial

```bash
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json --init /scratch/plan.json
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json --observe /scratch/baseline-observation.json
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json --select /scratch/selection.json
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json --observe /scratch/comparison.json
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json
python /path/to/skill/scripts/check.py experiment --state /scratch/current.json --export-best /scratch/restored-best
```

Initialization refuses an existing state. Selection replaces the current trial, explicitly abandoning any unfinished current trial; best is untouched. State holds only best/current source snapshots, the current hypothesis/result, evidence references and a selection counter. It does not append attempts, create dated directories or copy profiler captures. Updates use a lock and atomic replacement; reads do not rewrite state. Export requires a new directory and restores declared source bytes. The returned entry/configuration still determines how to invoke that snapshot. JSON output omits embedded source contents.

Declare every executable local dependency: entry, wrappers, helpers, dynamic imports, reference/comparator and relevant non-Python inputs. Installed libraries belong to toolchain identity. Dependencies are not inferred through imports. Source snapshots are bounded to 64 files and 2 MiB per variant and preserve relative paths; external dependencies must be represented in the toolchain record or explicitly copied into the declared root. Editing or deleting the original work files does not destroy best. Byte hashes include filenames, entry and configuration, so changing only a specialization also changes identity; that alone does not prove a structural optimization.

The selection count bounds this experiment state, not the task's device-request/time/token allowance. Exhaustion stops further selections in that state; completing current evidence and exporting best remain available. A self-chosen state cap does not authorize ending the user's task: preserve/export best and arrange the next working batch when actionable hypotheses remain, retaining cumulative consumption. A genuine user/platform hard limit still forbids further work covered by that limit; never reset it through another state. See [limit provenance](../workflows/optimization.md#budget-and-candidate-identity). Host execution quotas, device selection and per-command timeouts remain with the execution controller. Scope or toolchain changes require a new plan and validation. Source/toolchain drift and malformed evidence fail without rewriting state; state integrity hashes detect accidental manual edits.

### Plan and selection

`plan.json` freezes the original acceptance contract, representative timing cases, policy and baseline:

```json
{
  "schema": "pypto-experiment-plan/1",
  "toolchain": {"path": "/scratch/toolchain.json", "sha256": "<actual identity-file SHA256>"},
  "max_trials": 4,
  "scope": {
    "cases": {"rows8": {"rows": 8, "cols": 128, "workers": 1, "seeds": [0, 3]}, "rows32": {"rows": 32, "cols": 128, "workers": 1, "seeds": [0, 3]}},
    "timing_cases": ["rows32"],
    "metric": "native_task_sum_us",
    "numerical_contract": {"rtol": 0, "atol": 0, "oracle_sha256": "<actual independent reference SHA256>"}
  },
  "policy": {"min_pairs": 5, "min_improvement": 0.02, "min_win_fraction": 0.7},
  "baseline": {
    "root": "/scratch/source",
    "files": ["kernel.py", "reference.py", "driver.py"],
    "entry": "kernel.py:handover",
    "configuration": {"DIRECT": 0}
  }
}
```

The illustrative `max_trials=4` bounds one state file; it is not a recommended task-wide trial limit or evidence that four trials finish an optimization. Use the actual user/controller limit where one exists and keep local batching distinct from task completion.

The example policy requires a gain greater than 2% and at least 70% paired wins in each order, with at least five pairs per order. These configurable thresholds are not a statistical significance claim or universal acceptance criteria. Choose them against valid workload noise evidence and the task's protocol; the workflow's [sampling policy](../workflows/optimization.md#screen-candidates-before-confirmation) governs reuse and additional collection. The target metric is `native_task_sum_us` or `synchronized_wall_us`. Cases must specify the actual data regimes, launch geometry, input identity/generator and cache protocol relevant to the claim; the tool cannot discover omitted controls.

`selection.json` names the candidate snapshot and one testable hypothesis:

```json
{
  "schema": "pypto-experiment-selection/1",
  "variant": {"root": "/scratch/source", "files": ["kernel.py", "reference.py", "driver.py"], "entry": "kernel.py:handover", "configuration": {"DIRECT": 1}},
  "changes_wrapper": false,
  "hypothesis": {
    "observation": "The active aligned control converts ND to NZ before the L1 consumer.",
    "change": "Use the complete direct VF-to-NZ producer with the same consumer.",
    "prediction": "Remove one conversion; strided-store cost may offset it.",
    "reject_if": "Any numerical/guard failure, or no matched complete-call gain."
  }
}
```

`changes_wrapper` declares added or changed Host/wrapper work that native-task duration omits. Changed native task counts also trigger a paired wall-time gate. Do not mark it false to hide extra work. Static applicability and a hypothesis do not grant numerical acceptance.

### Controlled-verifier observation adapter

An execution controller can call `scripts/_checks/observations.py::capture` inside its existing verifier. It consumes the existing state, actual baseline/candidate snapshot specifications, frozen oracle artifact, an explicit `evidence_kind` (`native_measurement` or `synthetic_fixture`), and callbacks. `verify(side, case, workload)` returns `passed`, `failed` or `not_run` after using the original evaluator. When every numerical case passes, optional `measure(case, workload)` returns `{"timing": <unchanged compare result>}` and, when needed, `wall_timing` from the native timing tool.

The adapter binds the executed specifications to best/current, reads every declared source dependency before and after callbacks, checks the unchanged state/toolchain/oracle, and validates the projected observation through the existing experiment gates. It publishes one new observation file and returns its identity; it does not mutate state. Import that artifact with `experiment --observe`. A failed or unrun numerical case suppresses timing; callback errors propagate without publishing a success record. Existing evidence is never overwritten.

Callbacks own actual loading, execution, input/cache/worker matching and reference adequacy. They must load the declared bytes with ordinary source semantics and verify the actual native program. `evidence_kind` remains a declaration, not authentication. A generic command exit or model-authored JSON cannot supply the callbacks' numerical or timing acceptance. Execution controllers can invoke the adapter through an explicit `run` command; it adds no separate verifier.

### Verifier observation and retention

The verifier runs the actual variants, validates both against the frozen oracle and supplies its observation. To bind the executed files/configuration, import `_checks.experiment.variant` from the skill's scripts directory and compare its `sha256` before and after execution with the ID returned by `--init`/`--select`. Load the declared source bytes; a hash alone cannot detect a stale imported module or binary. Follow [compiled-variant verification](../verification/native-timing.md#verify-the-compiled-variant).

```json
{
  "schema": "pypto-experiment-observation/1",
  "scope": {},
  "toolchain_sha256": "<same verified identity-file SHA256>",
  "baseline": {"variant_sha256": "<executed baseline ID>", "correctness": {"scope": "complete_call", "oracle_sha256": "<frozen reference SHA256>", "cases": {"rows8": "passed", "rows32": "passed"}}},
  "candidate": {"variant_sha256": "<executed candidate ID>", "correctness": {"scope": "complete_call", "oracle_sha256": "<frozen reference SHA256>", "cases": {"rows8": "passed", "rows32": "passed"}}},
  "comparisons": [{"case": "rows32", "scope": "complete_call", "matched_inputs": true, "timing": {}}]
}
```

Replace `scope` with the exact plan scope and `timing` with the unchanged [`compare_native`/`compare_wall`](../verification/native-timing.md#compare-paired-implementations) result. Baseline-only verification omits `candidate` and `comparisons`. Case status is `passed`, `failed` or `not_run`; absent cases remain unverified. A subprocess exit code, parse success or a timing-only ablation cannot supply a numerical pass.

Observations are explicit external-verifier attestations, not independently replayed proofs. The checker validates their identities, coverage and internal consistency; it cannot establish the honesty of a custom verifier, the adequacy of its oracle or undeclared dependencies. Preserve the raw source-bound evidence in ignored scratch for review. It does not infer success from textual logs.

| Observation | State decision |
| --- | --- |
| Missing baseline acceptance | `baseline_unverified`; selecting a candidate is blocked |
| Either complete-call numerical result fails | `rejected`; best bytes stay intact; a failing baseline is marked unverified |
| Some acceptance cases are missing | `awaiting_correctness`; old best remains |
| A representative comparison or required wall comparison is missing | `awaiting_timing` |
| Both AB and BA pass the frozen improvement/win/sample policy on every timing case | `retained`; the candidate snapshot becomes best |
| Both orders regress on any timing case | `rejected`; do not silently accept a cross-shape tradeoff |
| Orders disagree, gain is within threshold or samples are insufficient | `inconclusive`; keep best and resolve the uncertainty or choose another hypothesis |

For a native-task target with wrapper/task-count changes, include `wall_timing` from `compare_wall` in each comparison; both metrics must pass the policy for automatic retention. This is a conservative automatic gate, not a general prohibition on measured tradeoffs. A different objective or a dispatch tradeoff needs its own explicit plan. Ratios are candidate/baseline, computed per pair by the timer; the tool validates counts, ranges and win consistency but does not reconstruct missing raw samples or assert confidence intervals.

The command's exit 0 means a valid state operation, including a rejected or inconclusive experiment. Inspect `phase`, `current.decision`, `best.verified` and `next_action`. Invalid schema/identity/path/metric returns exit 2 without promoting a candidate. A retained result applies only to this scope, representative timing subset, exact sources and toolchain; it establishes neither a hardware ceiling nor improved agent effectiveness.
