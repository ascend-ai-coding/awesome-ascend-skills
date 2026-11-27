---
name: pure-prefill-slo-tuning
description: Plan, execute, and report SLO-constrained pure-prefill (纯 P) tuning on Ascend using the user's model, image, dataset, benchmark conditions, and startup script. Use for topology, concurrency, and parameter search; not pure decode or ordinary service deployment.
---

# Ascend 纯 P SLO 寻优

Optimize **confirmed SLO-compliant QPS**, not an unconstrained peak or a short-sample winner. Read [references/workflow.md](references/workflow.md) for the complete execution method and [references/report.md](references/report.md) for the independent report format. This skill is self-contained and covers pure P only.

## Inputs and execution gate

Collect or verify the user-supplied **model/weights, image tag or digest, dataset and exact slice, baseline startup script, hardware/node, benchmark tool, effective request settings, and SLO constraints**. The request settings include temperature, `ignore_eos`, output length, shuffle/order, thinking mode, `rr`, timeout, prefix-cache policy, and the success rule. The user must specify or confirm the SLO metric and percentile (such as TTFT P50), threshold and comparison operator, required success rate, and resource ceiling. Do not infer these from a previous run; if missing, finish the read-only plan and obtain the values before execution.

Before live execution, inspect the script, image capabilities, dataset, and available hardware read-only. Present the user with:

1. A stage table covering baseline, fixed-parameter topology search, each topology's `cc` boundary, selected topology's parameter families, larger-sample confirmation, full-dataset confirmation, and final-service validation.
2. A **concrete startup-parameter table** showing the supplied script's actual value and planned candidate values/support checks for TP/DP/PCP, `--api-server-count`/`--headless` when applicable, max-num-batched-tokens, max-num-seqs, gpu-memory-utilization, block-size, HCCL_BUFFSIZE, SWA, prefix caching, speculative settings, communication, EPLB, and applicable extended knobs. Show exact script-copy locations, node substitutions, restart/rollback plan, `cc` ladder, `np` per point, and expected full-data run.

Wait for the user's confirmation of that **reviewable plan** before stopping a service, changing a live script, starting a variant, or benchmarking. Earlier explicit confirmation of the same plan counts; do not ask twice. If the user changes the workload, image, or load policy, revise the plan and remeasure a same-protocol baseline before ranking.

## Non-negotiable measurement rules

- In every pure-P `cc` ramp, use `cc = DP × k` for positive integer `k`, where DP is the number of data-parallel replicas, not API-server processes. Submit `np ≥ 4 × cc` prompts **at each point**. Limit `cc` to what the dataset supports; obtain an agreed repeated-data protocol before duplicating rows.
- Keep non-topology startup settings fixed during topology screening. For every valid topology, bracket and repeat its own SLO-compliant `cc` boundary. With `rr=inf`, `cc` is the load axis; finite-`rr` results cannot directly rank against it.
- Change one parameter family at a time on the selected topology, verify the feature actually took effect, and re-search its `cc` boundary. Cover all applicable families in the workflow; record unsupported, inactive, failed, and untested candidates honestly.
- Confirm `max-num-batched-tokens` on a larger sample and stable service before using it as the baseline for EPLB or other downstream parameter tests. If the chosen MBT or another baseline setting later changes, re-test affected shortlisted features on the final combination; results on the old baseline cannot justify including or excluding them.
- Preserve the original script and launcher with hashes. Use isolated copies and logs. Verify real long-input requests, cache policy, per-request success, TTFT, prefix-cache hit ratio, and service-side Running/Waiting. Do not stop another user's workload without authorization.
- Promote short scans to larger samples and the **entire specified dataset**. Confirm the actual final service, then deliver a standalone report using [references/report.md](references/report.md), including complete original/final startup script text and a portable reproduction recipe.
