# Create vLLM-Ascend Skill

## TL;DR

> **Quick Summary**: Create a comprehensive vllm-ascend skill for the awesome-ascend-skills repository, providing AI agents with knowledge to deploy and operate vLLM inference serving on Huawei Ascend NPU.
>
> **Deliverables**:
> - `vllm-ascend/SKILL.md` - Core skill documentation with quick start, deployment, quantization, distributed inference
> - `vllm-ascend/references/` - 6 detailed reference files
> - `vllm-ascend/scripts/` - 3 helper scripts
> - Updated `README.md` skills table
> - Updated `.claude-plugin/marketplace.json`
> - PR for repository contribution
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 7 → Task 8

---

## Context

### Original Request
Create a vllm-ascend skill based on the reference template from davila7/claude-code-templates, following skill-creator guidelines, and submit PR.

### Interview Summary
**Key Discussions**:
- Reference template: `davila7/claude-code-templates/inference-serving-vllm`
- Content scope: Comprehensive (6 reference files)
- Scripts: Standard set (3 scripts)
- Language: Bilingual (English + Chinese keywords)

**Research Findings**:
- vLLM-Ascend supports Atlas A2/A3/300I hardware
- Supports Qwen, DeepSeek, GLM models with W8A8/W4A8 quantization
- msmodelslim skill already covers quantization - vllm-ascend should focus on inference serving
- Reference template is NVIDIA-focused - need to adapt for Ascend-specific content

### Metis Review
**Identified Gaps** (addressed):
- Overlap with msmodelslim: vllm-ascend focuses on inference serving, links to msmodelslim for quantization
- NVIDIA-specific content: Remove GPTQ/AWQ/FP8, keep Ascend-specific (`--quantization ascend`)
- Cross-reference integrity: Use relative paths for all skill links

---

## Work Objectives

### Core Objective
Create a self-contained vllm-ascend skill that enables AI agents to deploy and operate vLLM inference serving on Huawei Ascend NPU, following existing skill patterns and conventions.

### Concrete Deliverables
- `vllm-ascend/SKILL.md` (≤500 lines)
- `vllm-ascend/references/installation.md`
- `vllm-ascend/references/deployment.md`
- `vllm-ascend/references/distributed.md`
- `vllm-ascend/references/performance.md`
- `vllm-ascend/references/troubleshooting.md`
- `vllm-ascend/references/quantization.md`
- `vllm-ascend/scripts/check_env.sh`
- `vllm-ascend/scripts/benchmark.py`
- `vllm-ascend/scripts/deploy_service.sh`
- Updated `README.md` skills table
- Updated `.claude-plugin/marketplace.json`

### Definition of Done
- [ ] All files created with proper content
- [ ] SKILL.md frontmatter includes bilingual keywords
- [ ] All internal links use relative paths
- [ ] marketplace.json passes JSON validation
- [ ] README.md includes new skill entry
- [ ] Changes committed and PR created

### Must Have
- SKILL.md with working Quick Start examples (offline inference + API server)
- Bilingual keywords in frontmatter
- Cross-references to related skills (msmodelslim, npu-smi, ascend-docker)
- Troubleshooting section with common issues

### Must NOT Have (Guardrails)
- NVIDIA-specific quantization methods (GPTQ, AWQ, FP8 on GPU)
- Duplicate quantization content from msmodelslim
- Generic vLLM content not relevant to Ascend
- Placeholder text in final deliverables

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: N/A (documentation only)
- **Automated tests**: N/A
- **Validation**: File existence, content validation, JSON syntax

### QA Policy
Every task includes agent-executed QA scenarios:
- **File creation**: Verify file exists with correct content
- **JSON validation**: `python -m json.tool marketplace.json`
- **Link validation**: Grep for relative path patterns
- **Frontmatter validation**: Verify YAML structure

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — scaffolding):
├── Task 1: Create vllm-ascend directory structure [quick]
├── Task 2: Create SKILL.md core content [writing]
└── Task 3: Create scripts (check_env.sh, benchmark.py, deploy_service.sh) [quick]

Wave 2 (After Wave 1 — references):
├── Task 4: Create references/installation.md [writing]
├── Task 5: Create references/deployment.md [writing]
├── Task 6: Create references/distributed.md [writing]
└── Task 7: Create references/performance.md [writing]

Wave 3 (After Wave 2 — remaining references):
├── Task 8: Create references/troubleshooting.md [writing]
└── Task 9: Create references/quantization.md [writing]

Wave 4 (After Wave 3 — integration):
├── Task 10: Update README.md skills table [quick]
├── Task 11: Update .claude-plugin/marketplace.json [quick]
└── Task 12: Create PR with changes [git]

Wave FINAL (After ALL tasks — verification):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Content quality review (unspecified-high)
└── Task F3: Cross-reference integrity check (deep)
```

### Dependency Matrix

- **1-3**: — — 4-9, 10, 11
- **4-9**: — — F1-F3
- **10-11**: 1-2 — 12
- **12**: 10, 11 — F1-F3

### Agent Dispatch Summary

- **Wave 1**: 3 tasks → T1 `quick`, T2 `writing`, T3 `quick`
- **Wave 2**: 4 tasks → All `writing`
- **Wave 3**: 2 tasks → All `writing`
- **Wave 4**: 3 tasks → T10-T11 `quick`, T12 `git`
- **Wave FINAL**: 3 tasks → F1 `oracle`, F2 `unspecified-high`, F3 `deep`

---

- [ ] 1. **Create vllm-ascend directory structure**

  **What to do**:
  - Create `vllm-ascend/` directory at repository root
  - Create `vllm-ascend/references/` subdirectory
  - Create `vllm-ascend/scripts/` subdirectory
  - Verify structure matches existing skills pattern

  **Must NOT do**:
  - Create nested subdirectories beyond references/ and scripts/
  - Create placeholder files (will be done in subsequent tasks)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple directory creation task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4-11
  - **Blocked By**: None

  **References**:
  - `npu-smi/` - Directory structure example
  - `msmodelslim/` - Directory structure example

  **Acceptance Criteria**:
  - [ ] `vllm-ascend/` directory exists
  - [ ] `vllm-ascend/references/` directory exists
  - [ ] `vllm-ascend/scripts/` directory exists

  **QA Scenarios**:
  ```
  Scenario: Directory structure created
    Tool: Bash
    Steps:
      1. ls -la vllm-ascend/
      2. ls -la vllm-ascend/references/
      3. ls -la vllm-ascend/scripts/
    Expected Result: All directories exist with drwxr-xr-x permissions
    Evidence: .sisyphus/evidence/task-1-structure.txt
  ```

  **Commit**: YES
  - Message: `feat(skills): add vllm-ascend skill directory structure`
  - Files: `vllm-ascend/`, `vllm-ascend/references/`, `vllm-ascend/scripts/`

- [ ] 2. **Create SKILL.md core content**

  **What to do**:
  - Create `vllm-ascend/SKILL.md` with comprehensive content
  - Include YAML frontmatter with:
    - `name: vllm-ascend` (MUST match directory name)
    - `description`: Comprehensive description for agent matching
    - `keywords`: Bilingual list (English + Chinese)
  - Include sections:
    - Quick Start (offline inference + API server examples)
    - Installation (Docker + pip methods)
    - Deployment (server mode + Python API)
    - Quantization (brief overview with link to msmodelslim)
    - Distributed Inference (tensor/pipeline parallelism)
    - Performance (key parameters)
    - Troubleshooting (common issues)
    - Scripts section
    - References section
    - Related Skills section
    - Official References
  - Keep SKILL.md ≤ 500 lines

  **Must NOT do**:
  - Include NVIDIA-specific content (GPTQ, AWQ on GPU)
  - Duplicate quantization content from msmodelslim
  - Use placeholder text

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation writing task
  - **Skills**: [`skill-creator`]
    - `skill-creator`: For skill creation guidelines and patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4-9
  - **Blocked By**: None

  **References**:
  - `npu-smi/SKILL.md` - Frontmatter and structure pattern
  - `msmodelslim/SKILL.md` - Comprehensive skill structure
  - `https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html` - Quick start examples
  - `https://docs.vllm.ai/projects/ascend/en/latest/installation.html` - Installation guide

  **Acceptance Criteria**:
  - [ ] SKILL.md exists with valid YAML frontmatter
  - [ ] `name` field equals `vllm-ascend`
  - [ ] `keywords` include both English and Chinese terms
  - [ ] Quick Start section has working code examples
  - [ ] All sections present
  - [ ] File length ≤ 500 lines
  - [ ] No placeholder text

  **QA Scenarios**:
  ```
  Scenario: SKILL.md structure valid
    Tool: Bash
    Steps:
      1. head -20 vllm-ascend/SKILL.md | grep "name: vllm-ascend"
      2. head -30 vllm-ascend/SKILL.md | grep "keywords:"
      3. wc -l vllm-ascend/SKILL.md
    Expected Result: name matches, keywords present, ≤ 500 lines
    Evidence: .sisyphus/evidence/task-2-skill-md.txt
  ```

  **Commit**: YES
  - Message: `docs(skills): add vllm-ascend SKILL.md`
  - Files: `vllm-ascend/SKILL.md`

- [ ] 3. **Create helper scripts**

  **What to do**:
  - Create `vllm-ascend/scripts/check_env.sh`: Verify CANN, torch-npu, vllm-ascend
  - Create `vllm-ascend/scripts/benchmark.py`: Performance benchmarking template
  - Create `vllm-ascend/scripts/deploy_service.sh`: Quick service deployment

  **Must NOT do**:
  - Hardcode paths (use variables/placeholders)
  - Create production-ready scripts (templates only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script creation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `msmodelslim/scripts/check_env.sh` - Environment check pattern
  - `atc-model-converter/scripts/check_env.sh` - Environment check pattern

  **Acceptance Criteria**:
  - [ ] `check_env.sh` exists and is executable
  - [ ] `benchmark.py` exists with proper argument parsing
  - [ ] `deploy_service.sh` exists and is executable

  **QA Scenarios**:
  ```
  Scenario: Scripts created
    Tool: Bash
    Steps:
      1. test -x vllm-ascend/scripts/check_env.sh && echo OK
      2. test -f vllm-ascend/scripts/benchmark.py && echo OK
      3. test -x vllm-ascend/scripts/deploy_service.sh && echo OK
    Expected Result: All scripts exist, shell scripts executable
    Evidence: .sisyphus/evidence/task-3-scripts.txt
  ```

  **Commit**: YES
  - Message: `feat(skills): add vllm-ascend helper scripts`
  - Files: `vllm-ascend/scripts/`
- [ ] 4. **Create references/installation.md**

  **What to do**:
  - Create `vllm-ascend/references/installation.md`
  - Include detailed installation guide:
    - Prerequisites (Python, CANN, torch-npu versions)
    - Docker installation method
    - pip installation method
    - Build from source method
    - Verify installation steps

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  - `https://docs.vllm.ai/projects/ascend/en/latest/installation.html`

  **Acceptance Criteria**:
  - [ ] File exists with comprehensive installation content
  - [ ] All installation methods documented

  **Commit**: YES (grouped with Tasks 5-9)

- [ ] 5. **Create references/deployment.md**

  **What to do**:
  - Create `vllm-ascend/references/deployment.md`
  - Include:
    - OpenAI-compatible server deployment
    - Python API usage
    - Configuration parameters table
    - Multi-node deployment

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocked By**: Task 1

  **Commit**: YES (grouped with Tasks 4, 6-9)

- [ ] 6. **Create references/distributed.md**

  **What to do**:
  - Create `vllm-ascend/references/distributed.md`
  - Include:
    - Tensor parallelism configuration
    - Pipeline parallelism configuration
    - Multi-node setup guide
    - Network verification steps

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocked By**: Task 1

  **Commit**: YES (grouped with Tasks 4-5, 7-9)

- [ ] 7. **Create references/performance.md**

  **What to do**:
  - Create `vllm-ascend/references/performance.md`
  - Include:
    - Key performance parameters table
    - Optimization recommendations
    - Benchmarking guide

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocked By**: Task 1

  **Commit**: YES (grouped with Tasks 4-6, 8-9)

- [ ] 8. **Create references/troubleshooting.md**

  **What to do**:
  - Create `vllm-ascend/references/troubleshooting.md`
  - Include:
    - Common errors and solutions
    - CANN version issues
    - Memory issues
    - Network issues

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 9)
  - **Blocked By**: Task 1

  **Commit**: YES (grouped with Tasks 4-7, 9)

- [ ] 9. **Create references/quantization.md**

  **What to do**:
  - Create `vllm-ascend/references/quantization.md`
  - Include:
    - Overview of quantization support
    - W8A8, W4A8, W4A4 details
    - Link to msmodelslim for quantization process
    - Deploy quantized models

  **Must NOT do**:
  - Duplicate msmodelslim content

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 8)
  - **Blocked By**: Task 1

  **Commit**: YES (grouped with Tasks 4-8)

- [ ] 10. **Update README.md skills table**

  **What to do**:
  - Add vllm-ascend entry to README.md skills table
  - Format: `| [vllm-ascend](vllm-ascend/SKILL.md) | vLLM inference engine for Ascend NPU: deployment, quantization, distributed inference |`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 11, 12)
  - **Blocked By**: Task 2

  **References**:
  - `README.md` - Existing skills table format

  **Acceptance Criteria**:
  - [ ] README.md includes vllm-ascend entry
  - [ ] Entry format matches existing skills

  **Commit**: YES
  - Message: `docs: update README with vllm-ascend skill`
  - Files: `README.md`

- [ ] 11. **Update .claude-plugin/marketplace.json**

  **What to do**:
  - Add vllm-ascend entry to marketplace.json plugins array
  - Format:
    ```json
    {
      "name": "vllm-ascend",
      "description": "vLLM inference engine for Huawei Ascend NPU. Deploy LLMs with OpenAI-compatible API, quantization (W8A8/W4A8), distributed inference (tensor/pipeline parallelism), and performance optimization. Supports Qwen, DeepSeek, GLM models.",
      "source": "./vllm-ascend",
      "category": "development"
    }
    ```

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 10, 12)
  - **Blocked By**: Task 2

  **References**:
  - `.claude-plugin/marketplace.json` - Existing structure

  **Acceptance Criteria**:
  - [ ] marketplace.json includes vllm-ascend entry
  - [ ] JSON is valid (passes syntax check)

  **QA Scenarios**:
  ```
  Scenario: JSON valid
    Tool: Bash
    Steps:
      1. python -m json.tool .claude-plugin/marketplace.json > /dev/null && echo OK
    Expected Result: No JSON parse errors
    Evidence: .sisyphus/evidence/task-11-json.txt
  ```

  **Commit**: YES
  - Message: `chore: register vllm-ascend in marketplace`
  - Files: `.claude-plugin/marketplace.json`

- [ ] 12. **Create PR with changes**

  **What to do**:
  - Create feature branch: `git checkout -b feature/add-vllm-ascend-skill`
  - Stage all changes: `git add vllm-ascend/ README.md .claude-plugin/marketplace.json`
  - Create commit: `git commit -m "feat(skills): add vllm-ascend skill for LLM inference on Ascend NPU"`
  - Push to remote: `git push origin feature/add-vllm-ascend-skill`
  - Create PR using `gh` CLI

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
    - `git-master`: For git operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after Tasks 10, 11)
  - **Blocked By**: Tasks 10, 11

  **Acceptance Criteria**:
  - [ ] Feature branch created
  - [ ] All changes committed
  - [ ] PR created and accessible

  **QA Scenarios**:
  ```
  Scenario: PR created
    Tool: Bash
    Steps:
      1. gh pr list --head feature/add-vllm-ascend-skill
    Expected Result: PR exists with correct title
    Evidence: .sisyphus/evidence/task-12-pr.txt
  ```

  **Commit**: NO (PR creation)

---

## Final Verification Wave (MANDATORY)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read plan end-to-end. Verify all "Must Have" present, all "Must NOT Have" absent. Check evidence files exist. Output: `Must Have [N/N] | Must NOT Have [N/N] | Files [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Content Quality Review** — `unspecified-high`
  Review all created files for: placeholder text, broken links, inconsistent formatting, missing frontmatter. Check SKILL.md ≤ 500 lines. Output: `Files [N clean/N issues] | VERDICT`

- [ ] F3. **Cross-Reference Integrity Check** — `deep`
  Verify all internal links use relative paths (`../skill/file.md`). Check marketplace.json category is valid. Verify skill name matches directory. Output: `Links [N/N valid] | Name match [YES/NO] | VERDICT`

---

## Commit Strategy

- **1**: `feat(skills): add vllm-ascend skill directory structure` — vllm-ascend/
- **2**: `docs(skills): add vllm-ascend SKILL.md` — vllm-ascend/SKILL.md
- **3**: `feat(skills): add vllm-ascend helper scripts` — vllm-ascend/scripts/
- **4-9**: `docs(skills): add vllm-ascend references` — vllm-ascend/references/
- **10**: `docs: update README with vllm-ascend skill` — README.md
- **11**: `chore: register vllm-ascend in marketplace` — .claude-plugin/marketplace.json

---

## Success Criteria

### Verification Commands
```bash
# Verify directory structure
ls -la vllm-ascend/
ls -la vllm-ascend/references/
ls -la vllm-ascend/scripts/

# Validate marketplace.json
python -m json.tool .claude-plugin/marketplace.json > /dev/null && echo "JSON valid"

# Check SKILL.md line count
wc -l vllm-ascend/SKILL.md

# Verify frontmatter
head -20 vllm-ascend/SKILL.md
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] SKILL.md ≤ 500 lines
- [ ] marketplace.json valid JSON
- [ ] All internal links use relative paths
- [ ] PR created and ready for review
