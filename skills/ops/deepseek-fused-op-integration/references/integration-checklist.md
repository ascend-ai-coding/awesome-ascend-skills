# Integration Checklist

## Algorithm
- [ ] Identify target model
- [ ] Locate original function
- [ ] Trace call chain
- [ ] Locate insertion point
- [ ] Identify training/inference differences

## Tensor Contract
- [ ] Input shapes
- [ ] Input dtypes
- [ ] Output shapes
- [ ] Output dtypes
- [ ] Tensor layouts
- [ ] Gradient requirements
- [ ] Parallelism constraints

## Integration
- [ ] Evaluate monkeypatch
- [ ] Evaluate direct replacement
- [ ] Select strategy
- [ ] Define feature flag
- [ ] Define fallback
- [ ] Inspect checkpoint compatibility
- [ ] Identify files to change

## Delivery
- [ ] Produce integration plan
- [ ] Record assumptions
- [ ] Record risks
- [ ] Provide rollback plan
