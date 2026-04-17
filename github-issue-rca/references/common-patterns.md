# Common Root Cause Patterns

This reference provides a catalog of common root cause patterns organized by issue type. Use this to help identify and validate hypotheses during root cause analysis.

## Error Type Patterns

### 1. Null Pointer / Undefined Errors

**Symptoms**:
- `NullPointerException`, `undefined is not a function`
- `Cannot read property X of undefined`
- Segmentation faults

**Common Root Causes**:
- Missing null/undefined checks
- Race condition in initialization
- Incorrect assumptions about data structure
- API contract violations (unexpected null response)

**Investigation**:
```bash
# Find null checks in affected area
grep -n "null\|undefined\|if\s*(!.*\)" path/to/file

# Check recent changes to initialization
git log -p -- path/to/file
```

### 2. Type Errors

**Symptoms**:
- `TypeError`, `ClassCastException`
- Unexpected behavior with certain inputs
- Property access failures

**Common Root Causes**:
- Missing type validation
- Incorrect type coercion
- API version mismatch
- Inconsistent data types in database/API

### 3. Resource Exhaustion

**Symptoms**:
- `OutOfMemoryError`, `ENOSPC`
- Timeouts, slow performance
- Connection failures

**Common Root Causes**:
- Memory leaks (unclosed resources, growing caches)
- Unbounded queries or data processing
- Missing connection pooling
- Inefficient algorithms

**Investigation**:
```bash
# Look for resource allocation without cleanup
grep -rn "new\|alloc\|connect\|open" path/to/module
grep -rn "close\|free\|disconnect\|release" path/to/module
```

### 4. Concurrency Issues

**Symptoms**:
- Intermittent failures
- Race conditions
- Deadlocks
- Data corruption

**Common Root Causes**:
- Missing locks or synchronization
- Incorrect lock ordering
- Non-atomic operations on shared state
- Event loop blocking (Node.js)

**Investigation**:
```bash
# Find shared mutable state
grep -rn "shared\|global\|static.*mutable" path/to/module

# Check synchronization patterns
grep -rn "lock\|mutex\|synchronized\|atomic" path/to/module
```

### 5. Network/API Failures

**Symptoms**:
- `ECONNREFUSED`, `ETIMEDOUT`
- 5xx errors, 4xx errors
- SSL/TLS errors

**Common Root Causes**:
- Service unavailability
- Network partitions
- Incorrect URLs/endpoints
- Authentication/authorization failures
- Rate limiting

**Investigation**:
```bash
# Check endpoint configurations
grep -rn "endpoint\|url\|host\|port" config/

# Look for timeout and retry logic
grep -rn "timeout\|retry\|backoff" path/to/module
```

### 6. Configuration Issues

**Symptoms**:
- Works in one environment but not another
- Unexpected feature behavior
- Missing functionality

**Common Root Causes**:
- Environment-specific configs
- Missing environment variables
- Config file not loaded
- Incorrect config precedence

**Investigation**:
```bash
# Check config file loading
grep -rn "config\|env\|settings" path/to/module

# Look for environment-specific code
grep -rn "process\.env\|getenv\|System\.getenv"
```

### 7. Database Issues

**Symptoms**:
- Query failures
- Slow queries
- Deadlocks
- Constraint violations

**Common Root Causes**:
- Missing indexes
- N+1 query problems
- Incorrect query construction
- Transaction scope issues
- Schema migration problems

### 8. Dependency Issues

**Symptoms**:
- Import errors
- Version conflicts
- Missing functionality
- Behavioral differences

**Common Root Causes**:
- Package version mismatch
- Missing dependencies
- Peer dependency conflicts
- Transitive dependency changes

**Investigation**:
```bash
# Check dependency versions
cat package.json | grep -A 20 "dependencies"
cat requirements.txt
cat go.mod

# Look for version-specific code
grep -rn "version\|require.*>" . | grep -i "package\|module"
```

## Pattern Recognition Heuristics

| Symptom Pattern | Likely Root Cause Category |
|-----------------|---------------------------|
| Intermittent, timing-related | Concurrency, race condition |
| Environment-specific | Configuration, dependencies |
| Gradual degradation | Resource leak, data growth |
| Sudden onset after deploy | Recent code change |
| Load-dependent | Performance, capacity |
| User-specific | Permissions, data issues |

## Investigation Priority Matrix

| Factor | Weight |
|--------|--------|
| Recent code changes in affected area | High |
| Direct error trace to specific line | High |
| Similar issues in the past | Medium |
| Environment/configuration differences | Medium |
| Dependency version changes | Medium |
| Load/timing factors | Low-Medium |

Use this matrix to prioritize which hypotheses to investigate first.
