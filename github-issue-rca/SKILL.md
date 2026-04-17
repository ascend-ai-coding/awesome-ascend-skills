---
name: github-issue-rca
description: GitHub Issue root cause analysis skill. Analyze specified GitHub issues to identify possible root causes by examining issue content, code repository, and related resources. Use when user asks to analyze issue root cause, investigate issue, RCA analysis, or troubleshoot GitHub issues. Provides investigation directions with probability estimates and explicitly states when root cause cannot be determined.
---

# GitHub Issue Root Cause Analysis

This skill provides a systematic approach to analyzing GitHub issues and identifying their possible root causes, with emphasis on searching similar issues first and collecting all relevant links.

## Output Requirement

**After completing any analysis, you MUST save the report to a markdown file:**
- File name: `{issue_number}_rca.md` (e.g., `123_rca.md` for issue #123)
- Location: Current working directory
- Tool: Use `write` tool to create/update the file
- If file doesn't exist: Create it directly with `write`
- If file exists: read it first, then overwrite with new content

**The analysis is considered incomplete until the report file is saved.**

---

## When to Use This Skill

Use this skill when:
- User asks to analyze a GitHub issue's root cause
- User wants to investigate or troubleshoot a GitHub issue
- User requests RCA (Root Cause Analysis) for an issue
- User provides a GitHub issue URL or issue number and asks for analysis

---

## Analysis Workflow

### Step 1: Gather Issue Information

1. **Extract issue details** using GitHub tools:
   - Issue title, description, and labels
   - Error messages, stack traces, or logs
   - Comments and discussions
   - Related commits or pull requests

2. **Identify affected components**:
   - File paths mentioned in error messages
   - Functions, classes, or modules referenced
   - Dependencies or external services involved

3. **Extract search keywords**:
   - Error message fragments
   - Function/class names
   - Key symptoms or behaviors

---

### Step 2: Search and Analyze Similar Issues (PRIORITY STEP)

**This is a PRIORITY step - perform this BEFORE diving into code analysis**

1. **Search for similar issues** using multiple strategies:
   ```
   # Strategy 1: By error message
   github_search_issues(q="error_fragment repo:owner/repo state:all")
   
   # Strategy 2: By symptom/behavior
   github_search_issues(q="symptom repo:owner/repo state:all")
   
   # Strategy 3: By file/component
   github_search_issues(q="filename repo:owner/repo state:all")
   
   # Strategy 4: By label
   github_search_issues(q="label:bug repo:owner/repo state:all")
   ```

2. **Identify relevant similar issues**:
   - Issues with same/similar error messages
   - Issues in same file or component
   - Issues with same symptom pattern
   - Recently closed issues (potential regression)

3. **Analyze similar issues in detail**:
   - **Root cause**: What was the root cause?
   - **Resolution**: How was it fixed? (PR link, commit)
   - **Pattern**: Is this a recurring issue?
   - **Related code**: Which files/functions were involved?

4. **Categorize similar issues**:
   | Category | Description | Value |
   |----------|-------------|-------|
   | Identical | Same error, same location | High - likely same root cause |
   | Similar | Same error, different location | Medium - same pattern |
   | Related | Different error, same component | Low - may share root cause |
   | Different | Unrelated issue | Skip |

5. **Extract insights from similar issues**:
   - Common root causes
   - Frequent problem areas
   - Recurring patterns
   - Previous fix approaches

6. **[MANDATORY] Collect all links**:
   - Issue URLs: `https://github.com/owner/repo/issues/NNN`
   - PR URLs: `https://github.com/owner/repo/pull/NNN`
   - Commit URLs: `https://github.com/owner/repo/commit/SHA`
   - File URLs with line numbers
   - These links MUST appear in the final report

---

### Step 3: Code Repository Analysis

1. **Locate relevant code**:
   - Use grep/search to find code related to error messages
   - Read files mentioned in stack traces
   - Examine recent commits in affected areas
   - Cross-reference with similar issues

2. **Analyze code patterns**:
   - Look for common bug patterns (null pointer, race condition, resource leak)
   - Check error handling and validation logic
   - Review configuration and environment-specific code
   - Compare with similar issue patterns

3. **[MANDATORY] Collect code links**:
   - File URLs: `https://github.com/owner/repo/blob/branch/path/to/file.py`
   - Line-specific: `https://github.com/owner/repo/blob/branch/path/to/file.py#L100`
   - Range URLs: `https://github.com/owner/repo/blob/branch/path/to/file.py#L100-L150`

---

### Step 4: External Context Investigation

1. **Check related resources**:
   - Documentation for affected features
   - Similar closed issues (search for patterns)
   - Recent changes or releases

2. **Use tools for additional context**:
   - GitHub: Query repository metadata, commits, PRs
   - Web search: Find known issues or discussions
   - Other tools: Based on issue domain

3. **[MANDATORY] Collect external links**:
   - Documentation URLs
   - Stack Overflow discussions
   - Blog posts or tutorials
   - Official guides or references

---

### Step 5: Cross-Reference Analysis with Similar Issues

**Compare current issue with similar issues identified in Step 2**

1. **Pattern matching**:
   - Is this a regression? (previously fixed, now broken)
   - Is this a known recurring issue pattern?
   - Did similar issues share the same root cause?

2. **Root cause correlation**:
   - If similar issue had root cause X, does current issue show same symptoms?
   - Are the code paths overlapping?
   - Is the same component/function involved?

3. **Adjust hypothesis confidence** based on similar issues:
   - High correlation: Boost confidence
   - Partial correlation: Consider combined root causes
   - No correlation: May indicate new issue

4. **Generate comparative analysis table**:

   | Similar Issue | Similarity | Root Cause | Status | Relevance |
   |---------------|------------|------------|--------|-----------|
   | #YYY | Identical | [Cause] | Closed | High |
   | #ZZZ | Similar | [Cause] | Open | Medium |

---

### Step 6: Root Cause Hypothesis Generation

Generate 2-5 hypotheses based on:

1. **Direct causes** (most obvious from error messages/code)
2. **Configuration issues** (environment, settings, versions)
3. **Integration issues** (APIs, services, dependencies)
4. **Timing/race conditions** (concurrency, async issues)
5. **Data issues** (corrupt data, missing data, invalid state)
6. **Regression from similar issues** (if pattern matches)

For each hypothesis, assess:
- **Evidence supporting**
- **Evidence against**
- **Confidence level** (High/Medium/Low)
- **Correlation with similar issues** (if applicable)

---

### Step 7: Generate Analysis Report and Save to File
**CRITICAL: After completing the analysis, save the findings to a markdown file. This is mandatory.**

**File**: `{issue_number}_rca.md` (e.g., `123_rca.md`)
**Location**: Current working directory

**The report MUST include ALL collected links in the References section.**

---

## Report Format (Simplified)

```markdown
# Root Cause Analysis: Issue #XXX

## Issue Summary
[Brief description]

## Similar Issues Analysis (PRIORITY)
| Issue | Similarity | Status | Root Cause |
|-------|------------|--------|------------|
| #YYY | Identical | Closed | [Cause] |

### Insights
- [Insight 1]
- [Insight 2]

## Possible Root Causes

### Hypothesis 1: [Name]
**Probability**: High/Medium/Low (%)
**Evidence**: [Supporting evidence]
**Similar Issue Correlation**: [Relation to similar issues]
**Recommended Investigation**: [Verification steps]

## References & Links
- Issue #YYY: https://github.com/owner/repo/issues/YYY
- PR #ABC: https://github.com/owner/repo/pull/ABC
- File: https://github.com/owner/repo/blob/main/path/to/file.py#L100
```

---

## Report Format (Detailed)

```markdown
# Root Cause Analysis: Issue #XXX

**Repository**: owner/repo
**Issue URL**: https://github.com/owner/repo/issues/XXX
**Analysis Date**: YYYY-MM-DD

---

## 1. Issue Summary

### Title
[Issue title]

### Description
[Brief summary]

### Labels
[List of labels]

### Key Symptoms
- Symptom 1
- Symptom 2

---

## 2. Similar Issues Analysis (PRIORITY SECTION)

### 2.1 Search Queries Used
- `error_fragment repo:owner/repo state:all`
- `symptom repo:owner/repo state:all`
- `filename repo:owner/repo state:all`

### 2.2 Similar Issues Found

| Issue | Title | Similarity | Status | Root Cause | Fix |
|-------|-------|------------|--------|------------|-----|
| #YYY | [Title] | Identical | Closed | [Cause] | [PR/Commit] |
| #ZZZ | [Title] | Similar | Open | Unknown | - |

### 2.3 Pattern Analysis
- **Is this a regression?**: Yes/No/Unclear
- **Recurring pattern**: [Description]
- **Previous fix approaches**: [How similar issues were resolved]

### 2.4 Insights from Similar Issues
1. **[Insight 1]** - Source: Issue #YYY
2. **[Insight 2]** - Source: Issue #ZZZ

---

## 3. Analysis Process
### Files Examined
- `path/to/file1.ext` - [Why examined]
- `path/to/file2.ext` - [Why examined]

### Code Areas Investigated
- [Function/Class/Module 1]
- [Function/Class/Module 2]

---

## 4. Possible Root Causes

### Hypothesis 1: [Name]
**Probability**: High/Medium/Low (XX%)

**Evidence Supporting**:
- [Evidence 1]
- [Evidence 2]

**Evidence Against**:
- [Counter-evidence]

**Correlation with Similar Issues**:
- Related to: Issue #YYY
- Pattern match: High/Medium/Low
- Regression candidate: Yes/No

**Root Cause Explanation**:
[Detailed explanation of how this could cause the observed issue]

**Recommended Verification Steps**:
1. [Step 1 - specific and actionable]
2. [Step 2 - specific and actionable]
3. [Step 3 - specific and actionable]

---

### Hypothesis 2: [Name]
[Same structure as Hypothesis 1]

---

### Hypothesis 3: [Name]
[Same structure as Hypothesis 1]

---

## 5. Investigation Priority
| Priority | Hypothesis | Rationale | Similar Issue |
| |----------|------------|-----------|------------------------|
| 1 | [Hypothesis name] | [Why this should be investigated first] | #YYY (if applicable) |
| 2 | [Hypothesis name] | [Why second] | #ZZZ (if applicable) |
    | 3 | [Hypothesis name] | [Why third] | - | |

---

## 6. Recommended Next Steps
### Immediate Actions
1. [Action 1]
2. [Action 2]

### Based on Similar Issues
- [If regression]: Check commit that introduced regression
- [If recurring]: Review why previous fix didn't prevent this
- [If new pattern]: Document as new issue type

### Additional Information Needed
- [Information that would help narrow down the cause]
- [Logs or data that should be collected]
### Potential Fix Directions
- [Direction 1 - possibly based on similar issue fix]
- [Direction 2]

---
## 7. Conclusion
[Summary of the most likely root cause and confidence]

**Key takeaway from similar issues**: [insights summary]

---
## 8. Unable to Determine Root Cause
**Status**: Unable to determine definitive root cause
**Reasons**:
- [Specific reason 1]
- [Specific reason 2]
**Additional Information Needed**:
- [What would help narrow down the cause]
---
## Appendix
### A. Error Messages/Stack Traces
```
[Paste relevant error messages or stack traces]
```
### B. Code Snippets
```[language]
[Relevant code snippets]
```
### C. Related Links & References
**⚠️ MANDATORY: Include ALL links discovered during analysis**

#### C.1 Related Issues
| Issue | Title | URL | Status | Relevance |
|-------|-------|-----|--------|-----------|
| #YYY | [Title] | https://github.com/owner/repo/issues/YYY | Closed | Identical issue, fixed in PR #ABC |
| #ZZZ | [Title] | https://github.com/owner/repo/issues/ZZZ | Open | Similar symptoms |
| #AAA | [Title] | https://github.com/owner/repo/issues/AAA | Closed | Same component issue |

#### C.2 Related PRs
| PR | Title | URL | Status | Relevance |
|----|-------|-----|--------|-----------|
| #ABC | [Title] | https://github.com/owner/repo/pull/ABC | Merged | Fix for issue #YYY |
| #DEF | [Title] | https://github.com/owner/repo/pull/DEF | Open | Related change |
#### C.3 Related Commits
| Commit | Message | URL | Date | Relevance |
|--------|---------|-----|------|-----------|
| abc123 | [Message] | https://github.com/owner/repo/commit/abc123 | YYYY-MM-DD | Introduced the bug |
| def456 | [Message] | https://github.com/owner/repo/commit/def456 | YYYY-MM-DD | Previous fix attempt |
#### C.4 Code References
| File | URL | Line(s) | Relevance |
|------|-----|---------|-----------|
| src/module/file.py | https://github.com/owner/repo/blob/main/src/module/file.py | 100-150 | Error origin |
| src/module/other.py | https://github.com/owner/repo/blob/main/src/module/other.py | 50-60 | Related logic |
#### C.5 Documentation & Guides
| Title | URL | Relevance |
|-------|-----|-----------|
| [Doc Title] | https://docs.example.com/... | [Why relevant] |
| [Guide Title] | https://guide.example.com/... | [Why relevant] |
#### C.6 External Resources
| Title | URL | Relevance |
|-------|-----|-----------|
| [Stack Overflow] | https://stackoverflow.com/... | [Why relevant] |
| [Blog Post] | https://blog.example.com/... | [Why relevant] |
#### C.7 Root Cause Case Library
**Link to similar resolved cases for reference:**
- Case 1: [Title] - https://github.com/owner/repo/issues/XXX (Same root cause)
- Case 2: [Title] - https://github.com/owner/repo/issues/YYY (Similar pattern)
#### C.8 All Search Queries Used
```
Query 1: "error_message repo:owner/repo state:all"
  → Results: X issues found
  → Relevant: #YYY, #ZZZ

Query 2: "symptom repo:owner/repo state:all"
  → Results: Y issues found
  → Relevant: #AAA
Query 3: "label:bug repo:owner/repo is:closed"
  → Results: Z issues found
  → Relevant: None
```
---
### D. Quick Reference Links Summary
**Most Important Links (Bookmark These):**
1. This Issue: https://github.com/owner/repo/issues/XXX
2. Most Similar Issue: https://github.com/owner/repo/issues/YYY
3. Previous Fix PR: https://github.com/owner/repo/pull/ABC
4. Suspect Code File: https://github.com/owner/repo/blob/main/path/to/file.py
5. Documentation: https://docs.example.com/...
```

---

## Tools to Use

- **GitHub**: `github_get_issue`, `github_search_issues`, `github_search_code`, `github_list_commits`
- **Code search**: `grep`, `glob`
- **File reading**: `read`
- **Web search**: For external context
- **Link construction**:
  - Issue: `https://github.com/{owner}/{repo}/issues/{number}`
  - PR: `https://github.com/{owner}/{repo}/pull/{number}`
  - Commit: `https://github.com/{owner}/{repo}/commit/{sha}`
  - File: `https://github.com/{owner}/{repo}/blob/{branch}/{path}#L{line}`

---

## Best Practices

1. **Search similar issues first** - This is the top priority before code analysis
2. **Use multiple search strategies** - Combine error, symptom, component searches
3. **Check for regressions** - Recently closed similar issues may indicate regression
4. **Start with error message** - Trace it back through the code
5. **Check recent changes** - Git blame/history for affected files
6. **Be explicit about uncertainty** - Better to say "unknown" than guess
7. **Provide actionable steps** - Each hypothesis should have verification steps
8. **[MANDATORY] Save the report** - Analysis is incomplete until saved to file
9. **[MANDATORY] Collect ALL links** - Every issue, PR, commit, file, doc link must be in the report
10. **Format links as clickable** - Use markdown: `[Title](URL)`
11. **Include line numbers** - Use `#L100` for specific lines

---

## Complete Workflow Example

1. User: "分析 GitHub issue #123"
2. Extract issue details using `github_get_issue`
3. **PRIORITY: Search similar issues** using `github_search_issues`
4. **Collect all relevant links** (issues, PRs, commits, files)
5. Analyze similar issues for patterns and fixes
6. Search code using `grep` and `github_search_code`
7. Review commits using `github_list_commits`
8. Read relevant files using `read`
9. Generate hypotheses with probability estimates
10. **Compile all discovered links**
11. **Write to `123_rca.md`** (must include all links in Appendix)
12. Inform: "分析完成，报告已保存到 123_rca.md"

---

## Link Collection Checklist

During analysis, track and output these link types:
- [ ] Related issues (from search results)
- [ ] Related PRs (from issue references or search)
- [ ] Related commits (from git history)
- [ ] Code files with line numbers
- [ ] Documentation links
- [ ] External resources (Stack Overflow, blogs, etc.)

**Every link should be formatted as a clickable markdown link in the report.**
