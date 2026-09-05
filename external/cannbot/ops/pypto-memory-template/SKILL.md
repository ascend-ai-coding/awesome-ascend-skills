---
name: external-cannbot-ops-pypto-memory-template
description: Template for the PyPTO Kernel operator-specific memory file (custom/{op}/MEMORY.md).
  Defines required sections, machine-readable fields, and update cadence. Use when
  creating or updating an operator's MEMORY.md.
original-name: pypto-memory-template
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Memory Template

This skill contains the memory template that agents copy to `custom/<operator_name>/MEMORY.md` at the start of Stage 5 (implementation). Stage 1–4 agents (planner / mathematician / architect / designer) do NOT read or write MEMORY.md — their deliverables live in SPEC.md, `<op>_golden.py`, DESIGN.md, and `module_interfaces.yaml`. MEMORY.md is created and owned by Stage 5 coder and downstream agents only.

## Contents

| File | Purpose |
|------|---------|
| **`MEMORY.template.md`** | The actual template — copy to `custom/<op>/MEMORY.md` and fill in |

---

## When to use

- **Stage 5 dispatch start:** Copy `MEMORY.template.md` to `custom/<operator_name>/MEMORY.md` as the first action; seed Task summary / API map from SPEC.md + API_REPORT.md, decomposition/contracts from DESIGN.md + `module_interfaces.yaml`
- **Every turn:** Update `active_module`, `modules_pypto_verified`, `current_staged_file`, `next_mandatory_step`
- **Stage 5 coding:** Fill in Golden function inventory, Module decomposition mirror, Module contracts, Staged module files table, Experience Preflight
- **Per-Phase:** Append to Per-module verification log after each boundary check
- **Cleanup:** Final Golden function inventory cross-check
- **Debugging:** Paste `extract_pypto_calls.py` output, append to Development & debug log

## Key sections in the template

| Section | When to fill | Mandatory? |
|---------|-------------|------------|
| Agent status (YAML) | Every turn (Stage 5+) | Yes |
| Task summary | Stage 5 start (from SPEC.md) | Yes |
| Validation | Stage 5 start | Yes |
| Module decomposition + rationale | Stage 5 start (mirror DESIGN.md/yaml) | Yes |
| Staged module files table | Stage 5 start (create), Per-Phase (update) | Yes |
| Per-module verification log | Each Phase M_k pass | Yes |
| API map | Stage 5 start (from API_REPORT.md) | Yes |
| Experience Preflight | Stage 5, before coding (coder creates) | Yes |
| Golden function inventory | Stage 5 coding (create), Per-Phase (cross-check) | Yes |
| Module contracts | Stage 5 start (mirror yaml) | Yes |
| Design format compliance | Stage 5 start | Yes |
| DEBUG_GUIDEBOOK.md §9 pre-write checklist | Before per-Phase coding | Yes |
| Development & debug log | Every error/fix | Yes |
| Human review milestones | Optional | No |
