# AGENTS.md - Guide for AI Coding Agents

This document provides coding guidelines for AI agents working on the Awesome Ascend Skills repository.

---

## Repository Overview

This is a **knowledge base repository** for Huawei Ascend NPU development, structured as flat AI Skills. Each skill is a self-contained directory with a `SKILL.md` file.

---

## Repository Structure

```
awesome-ascend-skills/
├── npu-smi/                    # npu-smi device management
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
├── hccl-test/                  # HCCL performance testing
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
├── ascendc/                    # AscendC Develop Helper
│   └── SKILL.md
└── README.md
```

---

## Commands

### Validation

```bash
# Verify all SKILL.md files have valid frontmatter
find . -name "SKILL.md" -exec head -5 {} \; -print

# Verify directory names match skill names
find . -name "SKILL.md" | while read f; do
  dir=$(dirname "$f")
  name=$(grep "^name:" "$f" | cut -d: -f2 | tr -d ' ')
  expected=$(basename "$dir")
  [ "$name" = "$expected" ] || echo "Mismatch: $dir has name='$name'"
done
```

---

## Code Style Guidelines

### SKILL.md Files (Required Format)

Every skill directory MUST contain a `SKILL.md` with this structure:

```yaml
---
name: skill-name                    # MUST match directory name
description: Clear description with keywords for agent matching.
---

# Skill Title

## Quick Start

Brief examples...

## Content Sections

Detailed instructions...

## Official References
- [Link text](url)
```

**Rules:**
- `name` MUST match the directory name exactly
- `description` MUST be comprehensive (for agent keyword matching)
- Use relative paths for internal links
- SKILL.md should be ≤ 500 lines (use references/ for detailed content)

### Progressive Disclosure

Keep SKILL.md lean:
- Core quick reference in SKILL.md
- Detailed documentation in references/
- Executable scripts in scripts/

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Directory names | `lowercase-with-hyphens` | `npu-smi`, `hccl-test` |
| Skill names | Match directory | `name: npu-smi` |
| Script files | `kebab-case.sh` or `snake_case.py` | `npu-health-check.sh` |

---

## Adding New Skills

1. Create directory at root level: `mkdir -p new-skill`
2. Create SKILL.md with proper frontmatter
3. Add references/ and scripts/ as needed
4. Update README.md skills table
5. **Update `.claude-plugin/marketplace.json`**: Add the new skill to the `plugins` array with appropriate category

---

## Key Principles

1. **Flat structure**: Skills live at root level, no nested hierarchies
2. **Independence**: Each skill should be usable independently
3. **Keywords**: Include relevant keywords in `description` for agent matching
4. **Progressive disclosure**: Core in SKILL.md, details in references/

---

## Official Documentation References

- Huawei Ascend: https://www.hiascend.com/document
- npu-smi: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html
