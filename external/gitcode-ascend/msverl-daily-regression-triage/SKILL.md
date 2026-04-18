---
name: external-gitcode-ascend-msverl-daily-regression-triage
description: Triage a daily msverl regression run by reading the baseline comparison
  log, stopping on success, extracting the most relevant training failure evidence
  from the daily training log when needed, collecting recent commits from verl main
  and MindSpeed master, and ranking the most likely culprit commits with concise fix-direction
  guidance.
original-name: msverl-daily-regression-triage
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# MSVerl Daily Regression Triage

Use this skill when a fixed daily `verl + MindSpeed` training job has run and Codex needs to decide whether the result is healthy, whether there is a training failure or an accuracy regression, and which recent commit is the most likely cause.

## Defaults

- Baseline comparison log: `/home/st_daily_verl/msverl.log`
- Training log pattern: `/home/st_daily_verl/logs/msverl_YYYYMMDD.log`
- `verl` repo: `https://github.com/verl-project/verl.git` on `main`
- `MindSpeed` repo: `https://gitcode.com/Ascend/MindSpeed.git` on `master`
- Cache root for temporary clones: `/tmp/msverl-skill-cache`
- Time window: from local previous day `00:00:00` to the task execution time

## Hard Stop Rules

- Read the comparison log first.
- If it contains `mean abs diff:` and the parsed value is exactly `0`, stop and report success.
- If it contains `mean abs diff:` and the value is non-zero, classify as `accuracy_regression`.
- If it contains `error, please check log`, classify as `train_error`.
- If the comparison log is ambiguous, report `unknown` and explain what evidence is missing before doing expensive work.

## Workflow

1. Run [parse_result_log.py](./scripts/parse_result_log.py) on the comparison log.
2. Stop immediately on `pass`.
3. For `train_error`, run [extract_failure_tail.py](./scripts/extract_failure_tail.py) against the daily training log and keep only the final high-signal error block.
4. For `accuracy_regression`, use the parsed reward lists and `mean abs diff` as the primary evidence.
5. Sync lightweight local clones with [sync_repos.py](./scripts/sync_repos.py).
6. Collect recent commits with [list_recent_commits.py](./scripts/list_recent_commits.py) for both repositories inside the default time window unless the user gives a different one.
7. Rank suspects with [rank_candidate_commits.py](./scripts/rank_candidate_commits.py).
8. Inspect diffs only for the top few commits when titles and touched files are not enough to explain a plausible fix direction.

## Cost Controls

- Never load the whole training log unless the tail-based extractor fails twice.
- Start with the log tail only; prefer the last traceback or last `ERROR` block.
- Rank commits using title and touched files before reading diffs.
- Limit deep diff reading to the top `3` candidates per repository unless the evidence is still weak.

## Expected Output

Return a compact report with:

- `status`: `pass`, `train_error`, `accuracy_regression`, or `unknown`
- `time_window`
- `evidence_summary`
- `candidate_repo`
- `candidate_commits`
- `confidence`: `high`, `medium`, or `low`
- `fix_direction`

When evidence is weak, say so clearly instead of forcing a single-commit claim.

## References

- Run [triage_msverl_regression.py](./scripts/triage_msverl_regression.py) for an end-to-end local workflow.
- Use [parse_result_log.py](./scripts/parse_result_log.py) and [extract_failure_tail.py](./scripts/extract_failure_tail.py) separately when validating logs by hand.
- Use [list_recent_commits.py](./scripts/list_recent_commits.py) when you need a raw recent-commit inventory without ranking.
