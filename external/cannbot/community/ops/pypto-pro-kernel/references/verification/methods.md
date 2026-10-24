# Working methods

[skill constraints](../../SKILL.md#constraints-and-workspace-policy) owns workspace constraints. Retain only methods that change the next action, not iteration histories, apologies, conversation narratives or copies of rules already owned by framework topics.

| Current activity | First action | Details |
| --- | --- | --- |
| Assessing API behavior or absence | Align the complete official page, installed signature/docstring and a minimal probe; search namespaces, name variants and both Python/C++ sides | [Evidence assessment](evidence.md) |
| Writing a kernel or probe | Check a same-family example and [constraint headings](../api/constraints.md), then use `check.py surface` to find a call |
| Writing references and assertions | Include positive/negative controls and control every byte read; do not invent an oracle for unknown behavior, and justify tolerances | [Evidence assessment](evidence.md) |
| Running on a device, diagnosing races or measuring performance | Read ignored environment notes, pin the toolchain, isolate dangerous cases and retain real exit codes | [Device runs](board-runs.md) |
| Changing diagnostics or validators | Assert user locations and the diagnostic body; remove the fix to confirm the negative case fails | [Diagnostic contracts](../diagnostics/messages.md) |

## Shell and editing

- Prefer `rg` for searches. BSD tools on macOS and GNU tools remotely may accept different options; use Python for complex text transformations.
- Quote glob-bearing zsh arguments, such as `-g '*.py'`. Store long temporary paths in variables and keep artifacts in ignored directories.
- Read actual text before bulk replacement; verify all intended matches are unique before writing. Review the diff afterwards rather than treating replacement counts as correctness evidence.
- Match syntax checks to the target Python version. A newer local interpreter accepting the file does not prove remote collection works; avoid relying on newer f-string syntax.
- For Markdown edits, check relative paths, anchors and tool-consumed tables as well as rendering.

A new method needs a real counterexample and an executable action. Prefer adding a test for a new instance of an existing method instead of creating another narrative document.
