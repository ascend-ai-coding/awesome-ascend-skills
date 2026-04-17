# Analysis Patterns

Common troubleshooting patterns and methodologies for issue analysis.

---

## Troubleshooting Methodologies

### 1. Binary Search Method (Bisection)

When issue appeared after unknown change:
```bash
git bisect start
git bisect bad HEAD
git bisect good <known_good_commit>
# Git guides through commits until root cause found
```

### 2. Divide and Conquer

Systematically narrow problem scope:
1. Isolate component (frontend/backend/database)
2. Isolate function/module
3. Isolate line of code

### 3. Working Backwards

Start from failure point:
1. What was the last successful state?
2. What changed between success and failure?
3. Trace the change through the system

### 4. Reproduce First

Before debugging:
1. Create minimal reproduction case
2. Document exact steps
3. Verify consistent reproduction

## Common Root Cause Patterns

### Configuration Issues

**Symptoms:**
- Works in one environment, fails in another
- Intermittent failures
- "It works on my machine"

**Debug:**
```bash
# Compare configs
diff config.prod.yaml config.dev.yaml

# Check environment variables
env | grep -i app

# Verify file paths
ls -la /path/to/config
```

### Concurrency Issues

**Symptoms:**
- Race conditions
- Heisenbugs (disappear when debugging)
- Timing-dependent failures

**Debug:**
- Add logging with timestamps
- Check for shared mutable state
- Look for missing locks/synchronization

### Memory Issues

**Symptoms:**
- Out of memory errors
- Performance degradation over time
- Crashes after long runtime

**Debug:**
```bash
# Check memory usage
free -h
top -o %MEM

# For Node.js
node --inspect app.js
# Chrome DevTools → Memory tab

# For Python
python -m tracemalloc script.py
```

### Dependency Issues

**Symptoms:**
- Version conflicts
- Missing packages
- Breaking changes after update

**Debug:**
```bash
# Check installed versions
npm ls package-name
pip show package-name

# Lock file inspection
cat package-lock.json | grep package
cat requirements.txt
```

### Network Issues

**Symptoms:**
- Timeouts
- Connection refused
- Intermittent connectivity

**Debug:**
```bash
# Test connectivity
ping host
curl -v https://endpoint
telnet host port

# DNS resolution
nslookup hostname
dig hostname

# Check ports
netstat -tlnp
lsof -i :port
```

## Issue Classification

### By Symptom Type

| Category | Keywords | Typical Causes |
|----------|----------|----------------|
| Crash | segfault, panic, OOM | Memory, null pointer, resource exhaustion |
| Performance | slow, timeout, lag | N+1 queries, memory leak, inefficient algo |
| Correctness | wrong result, unexpected | Logic error, type coercion, race condition |
| Availability | 500, connection refused | Service down, port conflict, firewall |
| Integration | API error, parse fail | Contract mismatch, version incompatibility |

### By Detection Stage

| Stage | Example | Prevention |
|-------|---------|------------|
| Compile | Syntax error, type error | CI, strict mode |
| Test | Unit test failure | TDD, coverage |
| Staging | Config mismatch | Environment parity |
| Production | User-reported bug | Monitoring, feature flags |

## Comment Analysis Patterns

### Key Phrases to Look For

**Root Cause Indicators:**
- "the problem is..."
- "this is caused by..."
- "root cause:"
- "because..."
- "turns out..."

**Solution Indicators:**
- "fixed by..."
- "the fix is..."
- "solved with..."
- "this works:"
- "PR: #"

**Debugging Steps:**
- "tried..."
- "tested..."
- "checked..."
- "can you..."
- "what if..."

### Timeline Reconstruction

From comments, build timeline:
```
T0: Issue reported
T1: First response / triage
T2: Investigation begins
T3: Hypothesis formed
T4: Hypothesis tested
T5: Root cause identified
T6: Fix proposed
T7: Fix merged
T8: Issue closed
```

## Metrics to Extract

### Time Metrics
- Time to first response
- Time to triage
- Time to resolution
- Number of interactions

### Complexity Metrics
- Number of participants
- Number of files changed
- Lines of code changed
- Number of related issues

### Quality Metrics
- Was reproduction provided?
- Was root cause identified?
- Was test added?
- Was documentation updated?

## Pattern Recognition

### Recurring Issues

Look for patterns across multiple issues:
- Same file/module frequently appears
- Same error message pattern
- Same component involved
- Same type of user affected

### Systemic Problems

When same root cause appears repeatedly:
- Missing tests in area
- Unclear documentation
- Complex/confusing API
- Technical debt hotspots
