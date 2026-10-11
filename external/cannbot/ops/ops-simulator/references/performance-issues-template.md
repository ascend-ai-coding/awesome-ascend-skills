# Performance Issue Documentation Template

Template and conventions for adding new entries to `performance-issues-{general,aic,aiv}.md`. Keep these files internally consistent so an agent can navigate them by the same rules in every file.

---

## When to add an issue here

| Symptom source | Target file |
|----------------|-------------|
| `imbalance_ratio`, load balance, core count, anything orthogonal to a specific pipeline | `performance-issues-general.md` |
| Bottleneck on an `AIC_*` pipeline (CUBE / MTE1 / MTE2 / FIXP / SCALAR on AI Cube core) | `performance-issues-aic.md` |
| Bottleneck on an `AIV*_*` pipeline (SIMD / MTE2 / MTE3 / SCALAR on AI Vector cores) | `performance-issues-aiv.md` |
| Cross-cutting issue triggered by *combination* of AIC + AIV signals (e.g. wasted Cube→Vector GM round-trip) | Put it in the file matching the **side that needs the code change**. The L0C→UB direct fix lives in `aic.md` because the Fixpipe change is on AIC. |

Do not create new files lightly. Only split out a new file when you have ≥ 3 issues that share a category and don't fit `general / aic / aiv`.

---

## Steps to add a new issue

1. **Pick the target file** using the table above.
2. **Pick the next free `§N` (or `§N.M` for sub-issue)** at the end of the file. Section numbers are greppable identifiers, not just visual structure — never reuse a number, never reorder existing ones.
3. **Add a row to the file's Quick Reference Table** with the issue name, detection metric path, threshold, fix headline, and the new `§N` pointer.
4. **Write the section body** following the structure below.
5. **If `SKILL.md`'s Quick Diagnosis Step 2 table needs an entry** (a new dominant pipeline → bound type mapping that didn't exist), add one row there. Otherwise leave `SKILL.md` alone.

---

## Required section structure

Every issue must include sections 1–3. Sections 4–6 are optional.

### 1. Problem Description (required)

One paragraph: what the pipeline / bottleneck is and why it shows up the way it does. No fix code here — just the failure mode.

### 2. Detection Criteria (required)

Table-only. Schema:

| Metric | Path | Condition |
|--------|------|-----------|
| `<short name>` | `<dot.path.in.summary.json>` | `<numeric threshold or qualitative cue>` |

**Always-present sections only**: `kernel_info`, `pipe_utilization`, `top_level_diagnosis`, `scalar_instructions`, `aiv_vector_instructions`, `pipeline_overlap`. These work on every CANN profiler build.

**If any row depends on `cache` or `bandwidth`** (which may be absent), add a second sub-table labelled **Fallback signal** that derives the same conclusion from only always-present fields. If no fallback exists, write one short paragraph explaining what the agent can do instead (e.g., re-profile, structural code inspection).

### 3. Fix Method (required)

Numbered or named subsections, one per root cause. Sub-headings must use the `§N.M` pattern so they're greppable from outside the file.

For each fix:
- One-line summary of what the fix changes
- Minimal C++ snippet showing `❌ before` vs `✅ after` if a code change is involved
- Prefer copy-pasteable code over prose
- Keep the entry **self-contained** — carry the minimal `❌/✅` snippet inline so the fix is usable without any external repo or network. **Do not** depend on links to `cann-samples` or other external repos (they may be unavailable to the user/agent), and do not copy a sample's narrative, diagrams, or walk-through. If a basic technique is owned by a sibling skill (double buffer, RegAPI, tiling, NPU arch), point to that **sibling skill** instead

### 4. Pitfalls (optional)

Bulleted list of "this looks plausible but breaks". Include things you discovered by debugging — these are high-value because nothing else captures them.

### 5. Verification after the fix (optional)

Short bulleted list of which metrics should move in which direction (qualitative only) so an agent can sanity-check whether the fix landed.

```markdown
**Verification after the fix**:
- `pipeline_overlap.<key pair>` should rise above <threshold>.
- `<dominant pipe>.mean` should drop / rise.
```

**Do not include measured benchmark numbers** (`kernel_total_clocks: 5651 → 4168`, `−13%`, etc.). They are tied to one kernel shape and one CANN build, age fast, and don't change the diagnosis or the fix. Keep the section qualitative; the agent runs the profile to get real numbers.

### 6. When Not to Apply (optional)

Bulleted list of scenarios where the fix is wrong / counterproductive (e.g. UB capacity exceeded, downstream kernel consumer instead of in-kernel AIV, the fix has already been applied).

### 7. Related Skills (optional)

```markdown
**Related Skills**:

📖 <short description>: [<skill-name>](https://gitcode.com/cann/cannbot-skills/blob/master/ops/<skill-name>/SKILL.md)
```

Only link to skills that the agent should read **as a follow-up**. Don't link upstream concept skills that are part of basic AscendC knowledge.

---

## Tone — what this skill is and isn't

- **Is**: diagnostic playbook. Each entry is `problem → detection → symptoms → fix`. An agent reads it to decide what to do next.
- **Isn't**: a tutorial or a benchmark report. The skill does **not** reproduce narrative explanations, ASCII pipeline diagrams, or before/after measurement tables from external samples. Carry only the minimal `❌/✅` snippet needed to apply the fix; keep entries self-contained (no dependency on external repos/links).

---

## Conventions

- **Threshold notation**: use backticks for code-ish values: `< 0.30`, `≥ 1.0`, `= 0.0`. Use ASCII operators (`<`, `>=`) or unicode (`≤`, `≥`) — pick one per file and stick with it.
- **Metric paths**: full dot-paths from the `summary.json` root. `pipe_utilization.pipeline_util_summary.AIC_CUBE.mean`, not just `AIC_CUBE.mean`.
- **Cross-references**: use `§N.M` (greppable) instead of markdown anchor links (fragile, encode the heading text).
- **No emoji in section IDs**. Emoji in body text is fine.
- **Code blocks**: always tag the language (`` ```cpp ``, `` ```bash ``, `` ```json ``).

---

## Minimal example

```markdown
## 7. <New Issue Name>

### Problem Description

<One paragraph.>

### Detection Criteria

| Metric | Path | Condition |
|--------|------|-----------|
| <name> | `pipe_utilization.pipeline_util_summary.AIVx_MTE2.mean` | `> 0.50` |

### Fix Method

#### §7.1 <Root cause name>

<One sentence summary.>

```cpp
// ❌ before
…

// ✅ after
…
```
```

After adding, update the Quick Reference Table at the top of the file with:

```markdown
| <Issue Name> | `<path>` | `<threshold>` | <one-line fix> | §7 |
```
