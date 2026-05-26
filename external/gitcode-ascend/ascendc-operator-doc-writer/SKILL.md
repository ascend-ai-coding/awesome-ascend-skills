---
name: external-gitcode-ascend-ascendc-operator-doc-writer
description: Write README-style technical documentation for AscendC custom operators
  by reading local source files and adapting an existing template. Use when Codex
  needs to document an AscendC operator, compare a target operator repo against a
  reference README, turn `op_host` and `op_kernel` implementations into structured
  docs, or generate per-operator documentation from code with the model instead of
  a parser script.
original-name: ascendc-operator-doc-writer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# AscendC Operator Doc Writer

## Overview

Use the model itself to read a reference README and the target AscendC source files, then synthesize a matching technical document. Do not rely on Python parser scripts to infer operator semantics; use shell search and read tools only to locate files and inspect source text.

## Workflow

### 1. Collect the template and target files

- Read exactly one reference README that represents the desired output format.
- Locate the operator implementation files by basename. In most AscendC projects this means:
  - one `op_host/<op_name>.cpp`
  - one `op_host/<op_name>_tiling.h`
  - one `op_kernel/<op_name>.cpp`
- If the target repo does not use a per-operator directory like the template repo, create a documentation path grouped by operator name, for example `.../docs/<op_name>/README.md`.
- When file ownership is ambiguous, inspect `CMakeLists.txt` or `rg` references before writing.

### 2. Build a source-backed understanding

- Extract confirmed facts from the host registration file:
  - operator name
  - inputs and outputs
  - dtypes and formats
  - platform gating such as `ascend910b`
- Extract tiling facts from the host and tiling header:
  - tiling fields
  - shape derivation
  - fixed constants
  - workspace formula
  - block and core split rules
- Extract kernel behavior from the kernel file:
  - runtime parameter parsing
  - main compute pipeline
  - mask behavior
  - reduction behavior
  - output writeback layout
- Treat anything not directly stated in code as an inference. Label it explicitly instead of presenting it as fact.

### 3. Write the README

- Mirror the template at the section level, but adapt the content to the target repo layout.
- Prefer a structure close to:
  - `Overview`
  - `Inputs And Outputs`
  - `Runtime Parameters`
  - `Tiling Design`
  - `Kernel Design`
  - `File Layout Mapping`
  - `Build Entry`
  - `Limitations`
  - `Summary`
- Keep the wording technical and source-backed.
- If the code does not provide runnable commands or a clear framework call site, say so instead of inventing one.
- If output shape or auxiliary outputs are implicit, explain that they follow the implementation contract and call out the uncertainty.

### 4. Verify before finishing

- Re-open the written README.
- Check every section against the source files that support it.
- Make sure the document distinguishes:
  - direct facts from code
  - reasonable inferences
  - unknowns that require upstream caller context
- Keep the final README concise enough to scan, but detailed enough that another engineer can understand how the operator is organized.

## Decision Rules

### If a template README exists

- Read it first and preserve its overall style.
- Reuse its section order when that order still fits the target operator.
- Rewrite the content from scratch based on the new source files; do not mechanically copy unrelated details.

### If the repo layout differs from the template repo

- Normalize the doc unit around the operator name rather than the folder shape.
- Add a short `File Layout Mapping` section that explains where host, tiling, kernel, and doc files live in the target repo.
- Prefer creating docs under a dedicated `docs/<op_name>/README.md` path when there is no natural per-operator root directory.

### If the user wants to use an external API

- Use the same workflow: gather the template and source files locally, then send a compact prompt plus the necessary excerpts to the API.
- Use the API as a writing engine, not as a parser replacement.
- Keep the prompt grounded in explicit file paths, extracted snippets, and required output sections.

## Resources

- See [references/readme-outline.md](references/readme-outline.md) for the default section checklist.
- See [references/source-review-checklist.md](references/source-review-checklist.md) for what to extract from host, tiling, and kernel files.
- See [references/model-prompt-template.md](references/model-prompt-template.md) for a prompt skeleton that works with either the current model or a user-provided API.

## Output Expectations

- Produce a README that another engineer can use without reopening all source files immediately.
- Prefer explicitly named sections over free-form prose.
- Cite uncertainty plainly when a fact is not derivable from the local code.
- Do not claim runtime behavior, call examples, or test coverage that the repo does not actually show.
