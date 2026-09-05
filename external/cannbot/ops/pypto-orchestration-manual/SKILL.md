---
name: external-cannbot-ops-pypto-orchestration-manual
description: pypto-op-orchestrator entry point for PyPTO Kernel development. Bundles
  the 3 control documents (principles, team roster, mandatory rules) as one skill
  with progressive-disclosure references. Read this file first, then load the references
  on demand. Use when orchestrating the PyPTO operator-development team.
original-name: pypto-orchestration-manual
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# pypto-op-orchestrator — PyPTO Kernel Development

This skill is the **entry point for the pypto-op-orchestrator** in the multi-agent
PyPTO kernel-development team. It contains the full operating manual for
the team as a set of progressive-disclosure references.

## Audience & scope (intentional exception)

> **⚠️ Orchestrator-only.** This skill — including all three referenced
> documents (`principles.md`, `agents.md`, `rules.md`)
> — is loaded **only by `pypto-op-orchestrator`**. No
> sub-agent (planner / mathematician / architect / designer / coder /
> verifier / debugger / optimizer) reads this skill, and no sub-agent
> Mandatory-reads list references any file under this directory.
>
> **Stage vocabulary is intentional here.** This skill is the
> **single canonical source of stage-management knowledge** —
> agent roster, dispatch decisions, gate criteria. Removing Stage
> vocabulary here would erase the orchestrator's operating manual.
>
> If you are a sub-agent and find yourself loading this skill, **stop**:
> you have been mis-dispatched. Hand control back to pypto-op-orchestrator.
>
> **Reading order:** Read this SKILL.md first, then `principles.md` →
> `agents.md` before the first dispatch. `agents.md` carries every
> sub-agent's inputs / deliverables / gate / handoff — enough to dispatch
> and gate without knowing how a sub-agent works. `rules.md` holds
> execution detail; load it only on demand, not to dispatch.

---

## References (Tier 3 — load on demand)

| # | Reference | Purpose | Load when |
|--:|-----------|---------|-----------|
| 1 | `references/principles.md` | 4 behavioral guidelines (Think, Simplify, Surgical, Goal-Driven) | Always load before the first dispatch of any session |
| 2 | `references/agents.md` | Sub-agent dispatch contract: roster (stage column), per-agent deliverables / gates / handoff, verifier modes, `failure_category` enum, debugger iteration cap. Sub-agent-internal skill loading lives in each `agents/<name>.md` Mandatory reads. | Load before the first sub-agent dispatch; keep it at hand across the whole session |
| 3 | `references/rules.md` | mandatory rules, module-at-a-time enforcement, 3 prohibitions. Every sub-agent output must pass these | On demand — sub-agents enforce these via their own skills |

These references were previously top-level files under `.agents/`. They
have been relocated here to enforce the skill-library structure: the pypto-op-orchestrator
Agent loads SKILL.md, then pulls in references only as needed.

---

## Shared state

All sub-agents read and write one shared file: `custom/<op>/MEMORY.md`. This
memory file is the single source of truth for stage status, gate evidence,
debug log, and module contracts. Every handoff between agents is a memory
update, not a direct message. Template:
skill `pypto-memory-template`'s `templates/MEMORY.template.md`.

---

