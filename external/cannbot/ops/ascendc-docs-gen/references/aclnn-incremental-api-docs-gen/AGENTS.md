# AGENTS.md

Python tool that generates **aclnn operator API documentation** by diffing two versions of CANN operator code (original vs modified) and producing/validating markdown docs via an LLM. Diffs come from comparing two checkouts (commit/tag/branch) of the same operator; the primary entrypoint is `src/run_all_cli.py`.

## Setup

- **Python 3.11.9** pinned via `.python-version` (3.11+ works). Install deps with pip: `pip install -r requirements.txt` — `requirements.txt` is the **sole dependency source** (loose constraints; no pyproject/lockfile).
- **Env vars** are read at a single site, `src/settings.py` (`load_settings()`, validated at entrypoint startup — never at import time):
  - `INCREMENTAL_LLM_KEY` — LLM API key (**required**; missing → exit 2 with the variable name in the message)
  - `INCREMENTAL_LLM_URL` — LLM base_url (**required**; missing → exit 2, no silent fallback to api.openai.com)
  - `INCREMENTAL_LLM_MODEL` — model name (optional; default `qwen3.8-max`, defined only in `src/settings.py`)
- Optional: `CANN_PLUGIN_LOG_DIR` (log dir; default `<repo>/data/logs`), `CANN_OPS_ROOT`, `CANN_OPS_WORKSPACE`, `CANN_EVIDENCE_DIR`, `TEST_OPERATOR_PATH`.
- `no_proxy` is **no longer set by the code** — if the LLM gateway must bypass a proxy, export `no_proxy` yourself (see `WORKFLOW.md`).

## Running — entrypoints

All real entrypoints live under `src/`.

**Primary — `src/run_all_cli.py`** (1377-line orchestrator; works both as `python src/run_all_cli.py` and `python -m src.run_all_cli` from the repo root). Authoritative CLI (verified against current argparse):
```
python src/run_all_cli.py \
  --original_op_repo <repo-root> --original_op_path <rel/op/path> \
  --modified_op_repo <repo-root> --modified_op_path <rel/op/path> \
  [--validate-only] [--skip-fix] [--md-stem <aclnnXxx> ...] \
  [--workspace_dir <dir>] [--output_data_path <dir>] [--dashboard]
```
- Flags omitted + TTY → interactive Rich/questionary TUI. Non-TTY (pipe/CI) + missing required flags (`--*_op_repo`/`--*_op_path`, and `--output_data_path`) → `parser.error` **exit 2** (no TUI fallback — headless-safe).
- `--md-stem` takes one or more markdown doc stems (e.g. `aclnnAbs aclnnAdd`); auto-selected if omitted. **This is NOT a version flag.**
- Default (no `--validate-only`) = full generate+validate pipeline; `--validate-only` = pre-validation only (skips change detection and generation); `--skip-fix` skips the fix step for failed validations.

**Stage / subprocess entrypoints** (emit `===...result===` stdout markers + `DONE`/`FAILED` files for the VSCode extension):
- `src/main_snapshot.py` — Stage 1 snapshot build only.
- `src/run_change_detection_pipeline.py` — Stages 2-6 (change detection + generation), non-interactive.
- `src/run_consistency_check_pipeline.py` — validate_only path, non-interactive.

**Other CLIs:**
- `python -m src.generation_validation.ui.cli` — interactive validation UI (`-f/--file`, `-d/--doc-part`, `-m/--mode`, `-l/--list`, `-o/--ops-dir`).
- `gitcode_pr_file_get.py <pr_url> [-o <dir>] [-t base head merge_base commit] [--sha <sha>]` — GitCode PR checkout (no token; public PRs; by-SHA fetch fallback for deleted source branches).

## Tests

```
python -m pytest         # preferred
# or: .venv/bin/python -m pytest tests/
```
- pytest 9.1.1; config in root `pytest.ini`: `pythonpath = .` (imports are uniformly `src.*`-absolute), `norecursedirs = .temp_git temp_git .temp_test_5824 comparison`. No pyproject.
- `norecursedirs` already excludes the vendored external repos (`.temp_git/`, `comparison/`) — plain `pytest` won't accidentally collect their tests. Never run `pytest` inside those dirs.
- conftest: `tests/anchor_construction/conftest.py` (fixtures `golden_dir`, `golden_data`, `path_config`, `data_loader`), `tests/snapshot_construction/conftest.py` (`sample_md_content`).
- **Mostly unit** (tmp_path + mocks, no network/git). Exceptions:
  - `tests/snapshot_construction/test_constraint_equivalence.py` — integration; needs `INCREMENTAL_LLM_*` env, else `pytest.skip()`.
  - `tests/dry_run_naming_validation.py` — **NOT a pytest test** (standalone script with own `main()`); run via `python tests/dry_run_naming_validation.py`, needs `CANN_OPS_ROOT` env pointing to a local ops-math checkout.
- `TEST_OPERATOR_PATH` env overrides the `path_config` fixture default (otherwise `<CANN_OPS_ROOT>/math/abs`); without either env the operator-dependent tests skip automatically.
- Tests don't hit the network; the pipeline CLIs do (`gitcode_pr_file_get.py` calls GitCode API + `git clone`).

## Architecture (src/)

Three subdirectories do the real work; two already have their own `AGENTS.md` — read those, don't duplicate:

| Module | Purpose | Entrypoint | Subdoc |
|---|---|---|---|
| `snapshot_construction/` | Build the "before" baseline: doc_json, code_map, tree_id, anchor/evidence tuples for all 5 doc_parts. | `snapshot.py` (`prepare_operator_data`, `load_snapshot`); CLI wrapper `src/main_snapshot.py`; anchor pipeline `anchor_construction/pipeline.py` (`AnchorPipeline`); tree_id `build_tree_id/run_pipeline.py` | — |
| `change_detection/` | 4-level anchor redirect (file→function→section→note) + doc-part change impact analysis. | `run_change_detection_cli.py` (`change_deteciton_cli()`), `anchor_redirect.py` (`run_anchor_redirect_pipeline`), `doc_part_change_detection/change_evaluator.py` (`run_evaluation`), `merge_evidence.py` | `src/change_detection/AGENTS.md` |
| `generation_validation/` | 4-stage LLM pipeline: Evidence→Style→Generation→Validation. | `ui/cli.py` (`run_full_pipeline`, `run_validate_only`) — called by ALL orchestrators; `main.py` (standalone) | `src/generation_validation/AGENTS.md` |
| `llm/` | `LLMClient` (OpenAI-compatible, retry/backoff). Leaf utility. | `client.py` (no `__main__`) | — |
| `md_fill/` | Fill generated content back into markdown by heading. Leaf utility. | `main_fill.py` (`update_md_with_dict`) | — |

**Three global singletons tie everything together** (the boundaries that matter):
- `snapshot_construction/pub_ref.py` → `PubRefConfig` (INCREMENTAL_LLM_KEY/URL/MODEL, DOC_PARTS, max workers). Imported across `change_detection/` and the top-level orchestrators — **the cross-module coupling point** (change_detection depends on snapshot_construction for config).
- `llm/client.py` → `LLMClient`. Imported by all three subdirs.
- `src/logger.py` → global logger (`setup_logger`, `info/debug/error/section/subsection/highlight/redirect_stdout_to_log`). Imported by ~every module.

**Deepest cross-module reuse:** `change_detection/anchor_build.py` imports 8 modules from `src.snapshot_construction` (cann_doc_struct, anchor_construction.constraint, summary_function_proto, support, build_tree_id, code_graph, sub_function, postprocess, common.*) — anchor rebuild re-runs snapshot_construction's anchor pipeline on the modified repo. `change_detection/merge_evidence.py` also imports `src.generation_validation.config` (`EVIDENCE_STAGES`).

**Packaging inconsistency:** `generation_validation/` uses explicit `__init__.py` packages with `__all__` re-export facades + relative imports. `change_detection/` and `snapshot_construction/` are namespace-package style (sparse `__init__.py`). Internal imports are **uniformly `src.*`-absolute**; a single `sys.path.insert(0, <repo-root>)` bootstrap line survives only in the 4 entrypoint scripts (`run_all_cli.py`, `main_snapshot.py`, `run_change_detection_pipeline.py`, `run_consistency_check_pipeline.py`) — the old sibling-import hacks (incl. the deleted `anchor_redirect_func/build_code_graph/`) are gone.

### Where to look
| Concern | Go to |
|---|---|
| Full pipeline orchestration | `src/run_all_cli.py` |
| Snapshot / baseline build | `src/main_snapshot.py` → `src/snapshot_construction/snapshot.py` |
| Anchor/evidence tuple building | `src/snapshot_construction/anchor_construction/pipeline.py` |
| Code graph / code_map | `src/snapshot_construction/code_graph/single_code_map.py` |
| tree_id / operator doc generation | `src/snapshot_construction/build_tree_id/run_pipeline.py` |
| 4-level change redirect | `src/change_detection/anchor_redirect.py` + `run_change_detection_cli.py` |
| Doc-part change impact | `src/change_detection/doc_part_change_detection/change_evaluator.py` |
| Evidence merge across levels | `src/change_detection/merge_evidence.py` |
| Generation & validation (LLM) | `src/generation_validation/ui/cli.py` |
| Consistency checking | `src/generation_validation/validator/consistency.py` |
| Markdown backfill | `src/md_fill/main_fill.py` |
| LLM client / global config | `src/llm/client.py`, `src/snapshot_construction/pub_ref.py` |
| Logging | `src/logger.py` |

## Repo layout & ignored dirs

`.gitignore` intentionally excludes: `.omo` (opencode/boulder agent state — **preserve, never commit**), `data` (all inputs/outputs/logs), `.diff`, `.claude/`, `.opencode/`, `out*`, `out_bak*`, `__pycache__/`. `.vscode/` is **untracked in practice** — the bare `.vscode` dir rule defeats the later `!.vscode/*.json` re-includes, so nothing under `.vscode/` is shared.

On-disk vendored external checkouts (both gitignored and excluded from pytest collection via `norecursedirs`) — do not edit or commit:
- `.temp_git/` — gitcode PR/commit checkouts cached by `run_pr_batch.py`/`gitcode_pr_file_get.py`.
- `comparison/` — vendored snapshot of `cann/ops-transformer` for diffing.

`resources/` (committed) holds the runtime data loaded by the generation-validation pipeline: `soul.json` (style/terminology constraints, loaded by `pipeline/stages.py`), `templates.yaml` (doc-part templates, `TemplateLoader`), `chip_map.yaml` (chip-to-product mapping, `config.py`).

## Optional peripheral tool — run_pr_batch.py (not the main flow)

The main flow is version-diff driven via `src/run_all_cli.py` (two checkouts of the same operator). `run_pr_batch.py` is an optional convenience wrapper for batch runs: it parses a user-created `gitcode_pr.txt` task list at the repo root (2 whitespace-separated columns `<pr_url>  <op_path>`; `#` comments and blank lines skipped; not committed), checks out each task's two versions via `gitcode_pr_file_get.py` into `.temp_git/<owner>_<repo>_<pr>/` (cached per run), then invokes `src/run_all_cli.py --original_op_repo <merge_base> --original_op_path <op_path> --modified_op_repo <head> --modified_op_path <op_path> --output_data_path data/outputs/snapshot/pr_<id>/<op_safe>` (full mode; `--md-stem` auto-selected downstream; optional `--dashboard` passthrough) → ANSI-stripped logs **and the batch timing/accuracy report** (`pr_batch_timing_<ts>.json` / `.md`) to `data/outputs/pr_batch_logs/` (same dir; `TIMING_OUTPUT_DIR = LOG_DIR`). Exits 1 if any task failed.

**No built-in data cleanup:** `.temp_git/` checkouts and `data/outputs/` artifacts/logs accumulate across runs (`--output_data_path` is an intermediate-artifact dir for troubleshooting only). Run in a temp directory or clean these up proactively after verifying deliverables. User-facing description: WORKFLOW.md §8 (附录).

## Conventions & gotchas

- **Hardcoded machine-specific paths** historically abound as argparse `__main__`/scratch defaults — they now read `CANN_OPS_ROOT` (or `TEST_OPERATOR_PATH`) env instead. Always pass `--workspace_dir`/`--*_op_repo` explicitly rather than relying on defaults.
- **`if __name__ == '__main__':` self-test blocks** are scattered through leaf modules (esp. `snapshot_construction/`) as dev scratch — don't mistake them for supported CLIs.
- **Date-stamped filename** `ancor_match_file_0422.py` (iterative dev artifact, not stable naming) holds the 3-part weighted file matching (40% code_map + 30% token + 30% LLM) — weights are hardcoded, not configurable.
- Entrypoint scripts keep a single `sys.path.insert(0, <repo-root>)` bootstrap line so `src.*` imports resolve in both script mode (`python src/...`) and `-m` mode (`python -m src....`) — run entrypoints from the repo root or via `python -m`.
- **CLI output markers:** subprocess entrypoints emit `===snapshot-result===` / `===change-detection-result===` / `==result==` stdout delimiters followed by JSON — the VSCode extension parses these. Preserve the marker strings if editing those entrypoints. The two stage entrypoints also clear stale `DONE`/`FAILED` sentinel files at startup (`_clear_sentinel_files`) and delete the opposite-state file before writing their own — a sentinel present after a run always belongs to that run.
- **QC skill:** for checking aclnn doc quality/conformance, use the `api-doc-qc-aclnn` skill.
