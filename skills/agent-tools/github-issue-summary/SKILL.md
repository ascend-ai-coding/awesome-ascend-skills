---
name: github-issue-summary
description: Analyze closed GitHub issues to create troubleshooting case studies with root cause analysis and lessons learned. Use when: (1) summarizing resolved issue troubleshooting process, (2) creating issue case studies, (3) documenting problem-solving methodology, (4) extracting lessons from closed issues, (5) building knowledge base from issue history. Triggers: "summarize this issue", "create case study from issue", "issue resolution summary", "troubleshooting case", "postmortem analysis".
---

# GitHub Issue Summary

Analyze closed GitHub issues and generate structured troubleshooting case studies documenting the problem-solving process, root cause, and lessons learned.

## Workflow

1. **Fetch Issue Data** — Retrieve issue details from GitHub
2. **Analyze Timeline** — Extract key events, comments, and resolution steps
3. **Generate Summary** — Create structured case study document
4. **Save Output** — Write to `docs/issue-cases/` (configurable)

## Quick Start

### Input: GitHub Issue URL or Number

```
https://github.com/owner/repo/issues/123
```

or

```
owner/repo#123
```

### Command to Fetch Issue

```bash
# Using GitHub CLI (recommended)
gh issue view 123 --repo owner/repo --json number,title,body,state,comments,labels,createdAt,closedAt,author

# Get all comments
gh api repos/owner/repo/issues/123/comments

# Get issue events (timeline)
gh api repos/owner/repo/issues/123/timeline
```

If `gh` CLI not available, use GitHub API directly:
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/owner/repo/issues/123
```

## Case Study Template

Generate case studies following this structure:

```markdown
# Issue #123: [Issue Title]

## Meta

| Field | Value |
|-------|-------|
| Repository | owner/repo |
| Status | Closed |
| Created | YYYY-MM-DD |
| Resolved | YYYY-MM-DD |
| Time to Resolve | X days |
| Labels | bug, priority-high |

## Problem Description

[Original issue body - describe the problem as reported]

## Root Cause Analysis

### Symptoms
- [Observable behavior that indicated the problem]

### Investigation Steps
1. [First debugging step taken]
2. [Second step...]
3. [Key discovery]

### Root Cause
[The actual underlying cause - be specific and technical]

## Resolution

### Fix Applied
[Description of the solution]

### Code/Configuration Changes
```[language]
// Relevant code snippet or config change
```

### Verification
[How the fix was verified to work]

## Lessons Learned

### Key Takeaways
1. [General principle learned]
2. [Pattern to watch for]
3. [Tool/technique that helped]

### Prevention Measures
- [ ] [Suggested preventive action]
- [ ] [Monitoring or alerting improvement]
- [ ] [Documentation update needed]

## Related Issues

- #456: Related issue
- #789: Similar problem

## References

- [Link to PR that fixed this]
- [Related documentation]
```

## Analysis Guidelines

### Identifying Key Information

1. **Problem Statement**
   - First comment (issue body)
   - Error messages, stack traces
   - Environment details (version, OS, config)

2. **Investigation Trail**
   - Follow-up comments with debugging attempts
   - Code snippets tested
   - Hypotheses raised and tested
   - Dead ends explored

3. **Resolution**
   - Final solution comment
   - Linked PR or commit
   - Confirmation of fix

### Key Comments to Extract

Look for comments containing:
- "root cause" / "caused by" / "because"
- "fixed" / "resolved" / "works now"
- Code blocks with solutions
- "try" / "test" / "check" (debugging attempts)
- Error outputs or logs

### Time Analysis

Calculate:
- Time from report to first response
- Time from report to resolution
- Number of back-and-forth exchanges

## Output Location

Default: `docs/issue-cases/issue-{number}-{slug}.md`

Where `slug` is a URL-safe version of the issue title:
```
"Fix memory leak in worker" → "fix-memory-leak-in-worker"
```

## Example Usage Scenarios

### Scenario 1: Summarize a specific issue

**User says:** "帮我总结这个issue的定位过程: https://github.com/xxx/yyy/issues/42"

**Actions:**
1. Fetch issue #42 data
2. Analyze comments for troubleshooting steps
3. Extract root cause and resolution
4. Generate case study markdown
5. Save to `docs/issue-cases/issue-42-{title-slug}.md`

### Scenario 2: Batch process recent issues

**User says:** "总结最近10个已关闭的bug类型issue"

**Actions:**
1. Query closed issues with `bug` label
2. For each issue, generate case study
3. Create index file linking all case studies

### Scenario 3: Extract common patterns

**User says:** "分析这些issue的共性问题和解决模式"

**Actions:**
1. Analyze multiple case studies
2. Identify recurring root causes
3. Summarize common debugging patterns
4. Generate pattern summary document

## GitHub CLI Reference

```bash
# View issue details
gh issue view <number> --repo <owner/repo>

# List closed issues
gh issue list --repo <owner/repo> --state closed --limit 20

# Get issue as JSON
gh issue view <number> --repo <owner/repo> --json number,title,body,state,comments,labels

# Get issue timeline
gh api repos/<owner>/<repo>/issues/<number>/timeline

# Search issues
gh issue list --repo <owner/repo> --search "error" --state closed
```

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /repos/{owner}/{repo}/issues/{number}` | Issue details |
| `GET /repos/{owner}/{repo}/issues/{number}/comments` | All comments |
| `GET /repos/{owner}/{repo}/issues/{number}/timeline` | Events timeline |
| `GET /repos/{owner}/{repo}/issues?state=closed` | List closed issues |

## Resources

- `scripts/fetch_issue.py` — Script to fetch issue data via GitHub API
- `references/case-template.md` — Full case study template
- `references/analysis-patterns.md` — Common troubleshooting patterns
