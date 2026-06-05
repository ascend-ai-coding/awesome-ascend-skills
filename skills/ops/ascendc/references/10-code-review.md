# Phase 10 — Code review (hypothesis-testing)

Security/compliance review of AscendC operator code against the coding red lines, using a
**hypothesis-testing** methodology. Every finding maps to a concrete rule clause; no
out-of-scope findings.

## Inputs

| Param | Required | Notes |
|---|---|---|
| Code | Yes | function / block / file to review |
| Review rule | Yes | what to check (e.g. "integer overflow", "memory leak") |
| Rule file path | Optional | explicit clause source; otherwise matched from the rule |

If any required input is missing, tell the user what is missing and stop.

## Rule categories

| Topic | Covers |
|---|---|
| 01 numeric operations | overflow, wrap-around, divide-by-zero, precision |
| 02 memory / pointer safety | out-of-bounds, null deref, GM/UB offsets |
| 03 resource management | leaks, Alloc/Free + EnQue/DeQue pairing |
| 04 input validation | shape/dtype checks, `TORCH_CHECK` |
| 05 concurrency safety | multi-core race, sync correctness |
| 06 operator interface | Runtime / Tiling / dynamic shape contracts |
| 07 ABI / interface compatibility | signature/schema stability |

## Process

### Stage 1 — Prepare
Validate inputs; read the matching rule clause(s); confirm the code segment.

### Stage 2 — Hypothesis testing (core)
1. **Segment** the code into independent units (functions / blocks / logical units).
2. For each segment set **H0 = safe**, **H1 = at risk**, confidence = 0%.
3. **Collect evidence** by dimension:

| Evidence type | Action | Score |
|---|---|---|
| red-line violation | match red-line clause | +40% |
| general-rule violation | match general clause | +20% |
| missing in-scope defense | check for defensive code in scope | +30% |
| call-chain risk | inspect called functions (LSP/Grep) | +25% |
| data-flow risk | trace variable origin / arithmetic | +25% |

4. **Validate evidence** (exclude false positives: defended elsewhere in scope; context
   proves it cannot trigger).
5. **Decide**: confidence = Σ valid evidence; **> 60% → flag as a risk** and report;
   otherwise move to the next segment.

Analysis requirements: use LSP for symbol defs and Grep for dependencies; for risky code,
check whether it's defended elsewhere in the file scope; follow function calls into their
bodies; inspect risky structs/members' definitions and arithmetic.

### Stage 3 — Report
For each finding show the hypothesis-testing trail (evidence chain + confidence) before
the detail. Report includes: review category; risk list (with line numbers + code
snippet); evidence chain; suggested fix. Verify reported line numbers are correct; cite
every risky block, not just one.

## Notes

- Read the full rule file first; count only what those rules cover.
- Use LSP/search for any call-chain that touches a suspect block.
- List uncertain items as "to confirm" for the user to judge.
- Keep cited code snippets short but clear.

## Anti-patterns (NEVER)

- NEVER report findings outside the provided rules.
- NEVER flag a risk without an evidence chain and confidence calculation.
- NEVER report a single line when multiple risky blocks exist.
