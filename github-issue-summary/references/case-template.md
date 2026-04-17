# Case Study Template

Full template for issue case studies. Copy and customize as needed.

---

# Issue #{number}: {title}

## Meta

| Field | Value |
|-------|-------|
| Repository | {owner}/{repo} |
| Issue URL | https://github.com/{owner}/{repo}/issues/{number} |
| Status | Closed |
| Author | @{author} |
| Created | {created_at} |
| Resolved | {closed_at} |
| Time to Resolve | {duration} |
| Labels | {labels} |
| Assignees | {assignees} |

## Problem Description

### Original Report

> {original_issue_body}

### Environment

- **Version/Branch**: {version}
- **OS/Platform**: {os}
- **Configuration**: {config}

### Expected Behavior

{what_should_have_happened}

### Actual Behavior

{what_actually_happened}

## Root Cause Analysis

### Symptoms Observed

- {symptom_1}
- {symptom_2}
- {symptom_3}

### Investigation Timeline

| Time | Action | Result |
|------|--------|--------|
| {t1} | {action_1} | {result_1} |
| {t2} | {action_2} | {result_2} |
| {t3} | {action_3} | {result_3} |

### Hypotheses Tested

1. **Hypothesis 1**: {description}
   - Test: {how_tested}
   - Result: {outcome}
   - Status: ❌ Rejected / ✅ Confirmed

2. **Hypothesis 2**: {description}
   - Test: {how_tested}
   - Result: {outcome}
   - Status: ❌ Rejected / ✅ Confirmed

### Root Cause

{detailed_technical_explanation_of_root_cause}

**Why it happened:**
{underlying_reason}

**Why it wasn't caught earlier:**
{missed_detection_reason}

## Resolution

### Solution

{description_of_fix}

### Code Changes

```{language}
// Before
{problematic_code}

// After
{fixed_code}
```

### Configuration Changes

```yaml
# Before
{old_config}

# After
{new_config}
```

### Related PR/Commit

- PR: #{pr_number}
- Commit: {commit_hash}
- Merged by: @{merger}

### Verification Steps

1. {verification_step_1}
2. {verification_step_2}
3. {verification_step_3}

## Lessons Learned

### What Worked Well

- {positive_1}
- {positive_2}

### What Could Be Improved

- {improvement_1}
- {improvement_2}

### Key Takeaways

1. **{takeaway_category_1}**: {lesson}
2. **{takeaway_category_2}**: {lesson}
3. **{takeaway_category_3}**: {lesson}

### Prevention Checklist

- [ ] Add unit test for {scenario}
- [ ] Update documentation for {topic}
- [ ] Add monitoring/alerting for {metric}
- [ ] Review similar code in {other_modules}
- [ ] Update runbook with {procedure}

## Related Issues

| Issue | Relationship | Notes |
|-------|--------------|-------|
| #{num} | {relationship_type} | {notes} |
| #{num} | {relationship_type} | {notes} |

## References

- [PR #{number}]({url})
- [Documentation]({url})
- [Related Discussion]({url})

---

## Template Shortcuts

### Minimal Template (Quick Summary)

```markdown
# Issue #{number}: {title}

**Problem**: {one_line_problem}
**Cause**: {one_line_cause}
**Fix**: {one_line_fix}
**Time**: {duration}

### Key Learning
{main_takeaway}
```

### Bug Fix Template

```markdown
# Bug: {title}

**Symptom**: {user_visible_symptom}
**Root Cause**: {technical_cause}
**Fix**: {solution}
**Prevention**: {how_to_prevent}
```

### Performance Issue Template

```markdown
# Performance: {title}

**Metric Affected**: {latency/throughput/memory}
**Before**: {baseline}
**After**: {improved}
**Improvement**: {percentage}

**Bottleneck**: {identified_bottleneck}
**Optimization**: {change_made}
```
