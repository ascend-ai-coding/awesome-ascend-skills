#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import pathlib
import sys
from zoneinfo import ZoneInfo


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_failure_tail import extract_error_block, read_tail  # noqa: E402
from list_recent_commits import list_recent_commits  # noqa: E402
from parse_result_log import parse_result_log  # noqa: E402
from rank_candidate_commits import rank_candidates  # noqa: E402
from sync_repos import REPOS, sync_repo  # noqa: E402


def default_training_log(run_time: dt.datetime, log_dir: str) -> pathlib.Path:
    return pathlib.Path(log_dir) / f"msverl_{run_time.strftime('%Y%m%d')}.log"


def compute_window(run_time: dt.datetime) -> tuple[str, str]:
    start_day = (run_time - dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (
        start_day.strftime("%Y-%m-%d %H:%M:%S %z"),
        run_time.strftime("%Y-%m-%d %H:%M:%S %z"),
    )


def build_fix_direction(issue_type: str, top_candidates: list[dict[str, object]]) -> str:
    if not top_candidates:
        return "No strong commit candidate was found. Re-check the log parser and widen the time window if needed."
    repo = top_candidates[0]["repo"]
    if issue_type == "train_error":
        return f"Inspect the top {repo} candidate first and verify whether the touched files changed error handling, distributed startup, or the failing execution path from the log tail."
    if issue_type == "accuracy_regression":
        return f"Inspect the top {repo} candidate first and verify whether the touched files changed reward computation, optimizer behavior, precision settings, or training data flow."
    return "Inspect the top candidate diff and compare it against the extracted evidence before widening the search."


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end triage workflow for the daily msverl regression job.")
    parser.add_argument("--result-log", default="/home/st_daily_verl/msverl.log", help="Comparison log path.")
    parser.add_argument("--training-log", help="Training log path. Defaults to /home/st_daily_verl/logs/msverl_YYYYMMDD.log.")
    parser.add_argument("--training-log-dir", default="/home/st_daily_verl/logs", help="Directory for daily training logs.")
    parser.add_argument("--cache-root", default="/tmp/msverl-skill-cache", help="Cache root for temporary repo clones.")
    parser.add_argument("--timezone", default="Asia/Shanghai", help="Timezone used for the default time window.")
    parser.add_argument("--run-time", help="Explicit run time in ISO format. Defaults to now in the selected timezone.")
    parser.add_argument("--limit", type=int, default=80, help="Maximum recent commits to inspect per repository.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of ranked commit candidates to keep.")
    parser.add_argument("--skip-sync", action="store_true", help="Skip git sync and assume cached repos already exist.")
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    run_time = dt.datetime.fromisoformat(args.run_time).astimezone(tz) if args.run_time else dt.datetime.now(tz)
    since, until = compute_window(run_time)

    result = parse_result_log(pathlib.Path(args.result_log))
    output: dict[str, object] = {
        "status": result["status"],
        "reason": result["reason"],
        "time_window": {"since": since, "until": until},
        "comparison": result,
    }

    if result["status"] == "pass":
        print(json.dumps(output, indent=2, ensure_ascii=True))
        return

    evidence_text = result["reason"]
    if result["status"] == "train_error":
        training_log = pathlib.Path(args.training_log) if args.training_log else default_training_log(run_time, args.training_log_dir)
        tail_lines = read_tail(training_log, 300)
        failure = extract_error_block(tail_lines, 40, 80)
        failure["path"] = str(training_log)
        output["failure_excerpt"] = failure
        evidence_text = failure["summary"]
    elif result["status"] == "accuracy_regression":
        evidence_text = f"mean abs diff {result['mean_abs_diff']}; latest rewards {result['latest_rewards']}; baseline rewards {result['baseline_rewards']}"

    cache_root = pathlib.Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_records = []
    repo_commits: dict[str, list[dict[str, object]]] = {}
    for repo_name, repo_cfg in REPOS.items():
        repo_dir = cache_root / repo_name
        if args.skip_sync and repo_dir.exists():
            repo_records.append(
                {
                    "name": repo_name,
                    "path": str(repo_dir),
                    "branch": repo_cfg["branch"],
                    "url": repo_cfg["url"],
                    "action": "reused",
                }
            )
        else:
            repo_records.append(sync_repo(cache_root, repo_name, repo_cfg))
        repo_commits[repo_name] = list_recent_commits(repo_dir, repo_cfg["branch"], since, until, args.limit)

    ranking_payload = {
        "issue_type": result["status"],
        "evidence_text": evidence_text,
        "repos": repo_commits,
    }
    ranking = rank_candidates(ranking_payload, args.top_k)
    output["repos"] = repo_records
    output["ranked_candidates"] = ranking["top_candidates"]
    output["fix_direction"] = build_fix_direction(result["status"], ranking["top_candidates"])

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
