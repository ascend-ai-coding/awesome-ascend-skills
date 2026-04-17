# AGENTS.md - Guide for AI Coding Agents

Coding guidelines for the Awesome Ascend Skills repository.

## Repository Overview

A **knowledge base repository** for Huawei Ascend NPU development, structured as flat AI Skills.

**Primary Skills:** npu-smi, hccl-test, atc-model-converter, ascend-docker, msmodelslim, ais-bench, vllm-ascend, ascendc, torch_npu, ascend-dmi, npu-docker-launcher, vllm-bench-serve, vllm-ascend-server, remote-server-guide

---

## Build/Test/Validate Commands

```bash
# Validate all skills
python3 scripts/validate_skills.py

# Validate single skill
python3 scripts/validate_skills.py | grep -A 3 "your-skill/"

# Quick checks
head -20 npu-smi/SKILL.md
grep "^name:" npu-smi/SKILL.md
```

**CI/CD:** `.github/workflows/validate-skills.yml` runs on push to `main`/`feat/**` and PRs.

---

## Repository Structure

```
awesome-ascend-skills/
├── npu-smi/                    # Skill directory
│   ├── SKILL.md                # Core content (≤500 lines)
│   ├── references/             # Detailed docs (optional)
│   └── scripts/                # Executable scripts (optional)
├── scripts/validate_skills.py  # Validation script
└── .claude-plugin/marketplace.json  # Plugin registry
```

---

## Code Style Guidelines

### SKILL.md Format

```yaml
---
name: skill-name                    # MUST match directory name
description: Clear description (≥20 chars)
keywords:                            # Optional
    - keyword1
---

# Skill Title

## Quick Start
Brief examples...

## Content Sections
Detailed instructions...
```

**Rules:**
- `name`: MUST match directory name exactly
- `description`: ≥20 characters for agent matching
- **Progressive disclosure:** Core in SKILL.md (≤500 lines), details in `references/`
- **Bilingual:** Chinese content encouraged
- **Code blocks:** Always specify language (```bash, python```)
- **Links:** Use relative paths

### Python Scripts
```python
#!/usr/bin/env python3
from typing import Dict, List, Tuple

def validate(path: str) -> Tuple[List[str], List[str]]:
    """Docstring."""
    return [], []
```

### Shell Scripts
```bash
#!/bin/bash
set -e
readonly DIR="$(cd "$(dirname "$0")" && pwd)"
```

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Directories | `lowercase-with-hyphens` | `npu-smi` |
| Skill names | Match directory | `name: npu-smi` |
| Scripts | `kebab-case.sh` / `snake_case.py` | `check-health.sh` |
| References | `lowercase-with-hyphens.md` | `queries.md` |
| Configs | `kebab-case.yaml` | `config.yaml` |

---

## Error Handling

**Python:** Explicit errors, actionable messages, `sys.exit(1)` on failure.

**Shell:** Use `set -e`, check return codes, provide fallback.

---

## Adding New Skills

1. `mkdir -p new-skill`
2. Create `SKILL.md` with frontmatter
3. Add `references/`, `scripts/`, `assets/` as needed
4. Update `.claude-plugin/marketplace.json`
5. Update `README.md` skills table
6. Run `python3 scripts/validate_skills.py`

**marketplace.json:**
```json
{
  "name": "new-skill",
  "description": "Description for agent matching",
  "source": "./new-skill",
  "category": "development"
}
```

---

## Key Principles

1. **Flat structure:** Skills at root level
2. **Independence:** Each skill usable independently
3. **Keywords:** Include in `description`
4. **Progressive disclosure:** Core in SKILL.md, details in `references/`
5. **Bilingual:** Chinese and English acceptable

---

## Validation Checklist

Before submitting PR:
- [ ] `name` matches directory name
- [ ] `description` ≥20 characters
- [ ] Valid YAML frontmatter
- [ ] Internal links resolve
- [ ] No `[TODO]` placeholders
- [ ] Added to `marketplace.json` and `README.md`
- [ ] `python3 scripts/validate_skills.py` passes

---

## References

- Huawei Ascend: https://www.hiascend.com/document
