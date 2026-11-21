# Coverage mode: enumerate, optimize and close

Use this mode only when the user asks to avoid missing known optimizations. Interactive optimization uses the [implementation loop](optimization.md#generate-a-concrete-alternative-from-the-computation).
Audited evidence is a separate target: this mode records closure, but it does not by
itself make a measurement reproducible.

Enumeration needs an explicitly selected card library. From this skill's root, select
the bundled library with `--cards-root references/knowledge-cards`, or pass another
library's path (see [tooling/optimization.md](../tooling/optimization.md)). The tools do
not discover a sibling directory. When no library is selected, coverage mode cannot
enumerate and must report coverage as incomplete instead of claiming closure.

## 1. Before enumerating

Freeze the task contract, baseline, workload set, metric and any supplied task limits. Use [limit provenance](optimization.md#budget-and-candidate-identity) for unspecified limits and local execution caps. For a requested audit trail, also freeze the evaluator and test hashes.

## 2. Enumerate

Read the configured card library's `index.md` once through the same `--cards-root`
source. Candidates are the `active` rows, not a reading sequence. Enumerate in this
order:

1. topic rows: `vec`, `cube`, `mem`, `pipe`, `scalar`, `sync`, `algo`, `recurrence`;
2. structural review ([structural-review.md](structural-review.md)), when triggered;
3. source-derived candidates, including opportunities from the initial structural review.

This order inventories candidates; it does not set execution priority or postpone a feasible structural experiment until every card closes. The [implementation loop](optimization.md#generate-a-concrete-alternative-from-the-computation) ranks comparisons by the actual computation and complete-call cost.

A card miss does not rule out a source-level optimization. If the source shows a
concrete mechanism that no card covers (adjacent fusible operations, an avoidable GM
round-trip, a redundant conversion), record it as a source-derived candidate with the
source evidence. Profiler evidence is preferred but not required for a source-level
candidate.

The explicitly selected card library is the enumeration source. Selecting a library
alone does not request coverage mode; the bundled library follows the same rule.

Cost control: if the user did not ask for coverage, do not run this mode.

## 3. Close each candidate

A row closes only as:

| Status | Evidence |
| --- | --- |
| `implemented` | code changed, full correctness, paired measurement, mechanism checked |
| `not_applicable` | source-level reason tied to this kernel |
| `blocked` | reproducible capability or contract block with a minimal case |
| `covered` | another row or an owner document already covers it, with a link |

"Expected gain is small", "profiler did not show it" and "no minimal reproduction" are
not closure reasons. A row whose lifecycle status is `capability-gated` closes as
`blocked` with the linked evidence; it is not an active candidate. Source-derived
candidates use the same closure states; a card miss is not a closure reason and does not
end the search.

## 4. Record

Write one line per candidate in the reply, or in the requested audit file:

```
id | applicability | expected metric | status | evidence
```

Do not add a ledger or a JSON report to the skill tree. For a requested audit trail,
write it under the project's `docs/perf/` location; that record does not by itself make
the measurement reproducible.

## 5. Measurement

Paired native-task timing remains the judge of "faster". Coverage closure is a process
result, not a performance result. A row being `implemented` does not prove a gain.

## 6. Stop

Stop when every enumerated row is in a terminal state and a fresh sweep finds no new candidate. If a verified task-wide limit expires first, identify its source and report unclosed rows and untested directions separately from hardware hypotheses. Sampling counts, command timeouts and local batch sizes do not by themselves close the task.
