# Example Analysis Output

## Example 1: Clear Root Cause Identified

```markdown
# Root Cause Analysis: Issue #1234

## Issue Summary
Application crashes with NullPointerException when processing user profile updates for users with missing email addresses.

## Analysis Process
1. Examined issue description and stack trace
2. Located affected code in `UserService.java:142`
3. Reviewed recent commits to the file
4. Checked similar closed issues

## Possible Root Causes

### Hypothesis 1: Missing Null Check for Email Field
**Probability**: High (85%)

**Evidence**:
- Stack trace points to `UserService.updateProfile()` line 142
- Code attempts to call `email.length()` without null check
- Issue occurs for "users with missing email addresses" (reported by user)
- Recent commit 3 days ago removed validation middleware

**Counter-evidence**:
- None identified

**Recommended Investigation**:
1. Verify line 142 in `UserService.java` lacks null check
2. Check git history for validation middleware removal
3. Add null check and test with users missing email
4. Consider restoring validation at middleware level

### Hypothesis 2: Database Data Inconsistency
**Probability**: Low (15%)

**Evidence**:
- Some users in database may have null email fields

**Counter-evidence**:
- Code should handle null regardless of data state
- Not all users with null emails trigger the issue (timing-related)

**Recommended Investigation**:
1. Query database for users with null/empty emails
2. Add data validation constraints

## Investigation Priority
1. **Add null check in `UserService.updateProfile()`** - Direct fix for the crash
2. **Restore input validation** - Prevent future similar issues
3. **Database cleanup** - Ensure data consistency

## Next Steps
- [ ] Add null check at line 142
- [ ] Add unit test for null email case
- [ ] Review validation middleware removal decision
```

## Example 2: Multiple Possible Causes

```markdown
# Root Cause Analysis: Issue #567

## Issue Summary
Intermittent API timeouts affecting 5-10% of requests in production environment only.

## Analysis Process
1. Reviewed error logs and timing data
2. Examined database query patterns
3. Checked network configuration and timeouts
4. Analyzed recent deployment changes

## Possible Root Causes

### Hypothesis 1: Database Connection Pool Exhaustion
**Probability**: Medium (50%)

**Evidence**:
- Timeouts correlate with high traffic periods
- Database query duration is within normal range
- Similar issue occurred 6 months ago (Issue #234)

**Counter-evidence**:
- Connection pool metrics show available connections during timeouts
- No database errors in logs

**Recommended Investigation**:
1. Review connection pool configuration
2. Check for connection leaks
3. Monitor active connections during incident

### Hypothesis 2: Network Latency Spike
**Probability**: Medium (35%)

**Evidence**:
- Production-only issue (staging doesn't have same network topology)
- Timeouts are intermittent, suggesting transient network issues

**Counter-evidence**:
- No network monitoring alerts during incidents
- Other services on same network unaffected

**Recommended Investigation**:
1. Review network monitoring during incident window
2. Check DNS resolution times
3. Trace network path for affected requests

### Hypothesis 3: Upstream Service Degradation
**Probability**: Low (15%)

**Evidence**:
- API depends on external payment service
- Payment service SLA doesn't guarantee 100% uptime

**Counter-evidence**:
- No timeout correlation with payment request patterns
- Error rate doesn't match payment service status page

**Recommended Investigation**:
1. Correlate timeouts with payment service calls
2. Add circuit breaker for payment service
3. Review retry logic

## Investigation Priority
1. **Database connection pool analysis** - Most probable based on history
2. **Network path analysis** - Production-specific suggests infrastructure
3. **External service correlation** - Rule out or confirm dependency

## Cannot Determine Root Cause
The available information is insufficient to determine the definitive root cause. Multiple hypotheses have similar evidence levels. Additional monitoring and logging during incidents is needed to narrow down the cause.

**Additional information needed**:
- Connection pool metrics during incidents
- Network latency measurements
- External service call correlation
- Thread dump during timeout
```

## Example 3: Unable to Determine Root Cause

```markdown
# Root Cause Analysis: Issue #789

## Issue Summary
User reports data inconsistency where calculated totals don't match item sums, but issue is not reproducible by developers.

## Analysis Process
1. Examined reported data discrepancy
2. Reviewed calculation code
3. Checked for race conditions in data flow
4. Searched for similar issues

## Possible Root Causes

### Hypothesis 1: Browser Cache Serving Stale Data
**Probability**: Low (20%)

**Evidence**:
- User-specific issue, not reproducible elsewhere
- Could explain stale data display

**Counter-evidence**:
- User reports clearing cache didn't help
- Issue persists across sessions

### Hypothesis 2: Concurrent Edit Race Condition
**Probability**: Medium (40%)

**Evidence**:
- Calculations involve multiple database tables
- No explicit locking mechanism in code
- Intermittent nature suggests timing factor

**Counter-evidence**:
- Unable to reproduce with concurrent edit tests
- No evidence of concurrent access in logs

## Unable to Determine Definitive Root Cause

**Reasons**:
1. **Cannot reproduce the issue** - Issue only occurs for specific user under unknown conditions
2. **Insufficient logging** - Current logs don't capture calculation input states
3. **Unknown user actions** - Don't know the exact sequence of actions that led to the discrepancy
4. **Timing information missing** - Don't know when the discrepancy first appeared

**Additional information needed**:
1. User's exact action sequence before noticing discrepancy
2. Browser and version information
3. Screenshot or export of inconsistent data
4. Timestamp when discrepancy was first observed
5. Add detailed logging to calculation process for future incidents

## Recommended Actions
1. **Add comprehensive logging** to capture calculation inputs and outputs
2. **Add data validation** to detect inconsistencies early
3. **Consider adding optimistic locking** for concurrent edits
4. **Request more information** from user (screenshots, browser info, action sequence)
5. **Monitor for recurrence** with enhanced logging enabled

## Monitoring Recommendations
```javascript
// Add logging to capture calculation details
logger.info('Calculation input', {
  userId,
  items: items.map(i => ({ id: i.id, amount: i.amount })),
  timestamp: Date.now()
});
logger.info('Calculation result', { userId, total, expectedSum, diff });
```
```
