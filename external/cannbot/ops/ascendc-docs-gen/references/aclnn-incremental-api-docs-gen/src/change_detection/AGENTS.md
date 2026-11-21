# CHANGE DETECTION

**Generated:** 2026-05-27

## OVERVIEW
Multi-level anchor redirect for change detection between CANN operator versions.

## STRUCTURE
```
change_detection/
├── run_change_detection_cli.py    # change_deteciton_cli() — called (imported) by run_all_cli.py
│                                  #   and run_change_detection_pipeline.py
│                                  #   (run_change_detection.py deleted — superseded, zero refs)
├── anchor_build.py                # Anchor construction/rebuild (top level)
├── anchor_redirect.py             # 4-level redirect pipeline (run_anchor_redirect_pipeline)
├── anchor_redirect_file/          # Level 1: file-level matching
│   ├── ancor_match_file_0422.py   # 3-part weighted matching
│   └── unit_config.py
├── anchor_redirect_func/          # Level 2: function-level matching
│   ├── function_redirect_v1.py    # process() — function redirect
│   ├── three_level_mapper.py      # ThreeLevelMapper
│   ├── config.py / utils.py
│   │                              #   (build_code_graph/ deleted — orphaned, zero refs)
├── anchor_redirect_section/       # Level 3: anchor_retarget_section.py
├── anchor_redirect_note/          # Level 4: anchor_retarget_note.py
├── merge_evidence.py              # Merge extracted evidence across levels
├── doc_part_change_detection/     # Doc-part level change impact analysis
│   ├── change_evaluator.py        # LLM-based change importance filter
│   ├── file_change_detector.py    # File diff detection (MD5 + unified_diff)
│   ├── subpart_mapper.py          # Map changed files → affected doc subparts
│   └── docpart_file_extractor.py  # Extract docpart↔file mappings from evidence JSON
└── test_*.py                      # Dev self-test scripts (not supported CLIs)
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Run full detection | `run_change_detection_cli.py` | `change_deteciton_cli()` — invoked by the orchestrators; standalone stage entrypoint is `src/run_change_detection_pipeline.py` |
| File-level redirect | `anchor_redirect_file/ancor_match_file_0422.py` | Part1 (path) → Part2 (same-dir) → Part3 (LLM), weighted scoring |
| Build anchors | `anchor_build.py` | Constructs anchor objects from matched pairs |
| Function-level redirect | `anchor_redirect_func/function_redirect_v1.py` | Maps functions across versions (build_code_graph/ was deleted — the code-graph twin lives in `src/snapshot_construction/code_graph/`) |
| Merge evidence | `merge_evidence.py` | Combines evidence from all 4 redirect levels |
| Parse CLI output | Look for `===change-detection-result===` marker | JSON follows this stdout delimiter |

## CONVENTIONS
- **4-level cascade**: file → function → section → note, each level narrows scope
- **3-part file matching**: Part1 (path match, 40% code_map + 30% token + 30% LLM), Part2 (same-directory fallback), Part3 (LLM-guided)
- **No `__init__.py`**: Namespace package style (differs from generation_validation which uses explicit packages)
- **Cross-module LLM config**: Imports `PubRefConfig` from `src.snapshot_construction/pub_ref.py`; credentials are injected at entrypoint startup via `src/settings.py` (`load_settings()` + `apply_settings`)
- **Verb-prefix naming**: `extract_*`, `merge_*`, `fill_*`, `step_N_*` at depth 4+
- **Self-test blocks**: `if __name__ == '__main__':` for local validation

## ANTI-PATTERNS
- **No `__init__.py`**: Namespace package style retained (differs from generation_validation which uses explicit packages) — safe now that all imports are `src.*`-absolute; the old sibling `sys.path.insert` hacks (deleted `build_code_graph/single_code_map.py`) are gone
- **Hardcoded weights**: 40/30/30 scoring split in `ancor_match_file_0422.py` not configurable
- **Date-stamped filename**: `ancor_match_file_0422.py` suggests iterative development artifact, not stable naming
- **Cross-module dependency**: Relies on `snapshot_construction/pub_ref.py` singleton, tight coupling between pipeline stages
