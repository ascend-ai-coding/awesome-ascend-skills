# Operator Validation Checklist

## Environment
- [ ] PyTorch version
- [ ] torch_npu version
- [ ] CANN version
- [ ] NPU model
- [ ] Operator package availability

## Functional Tests
- [ ] Operator import
- [ ] Forward execution
- [ ] Input shape checks
- [ ] Output shape checks
- [ ] Dtype checks
- [ ] Reference correctness comparison
- [ ] Gradient comparison
- [ ] Boundary cases

## Integration Tests
- [ ] Fused mode enabled
- [ ] Fused mode disabled
- [ ] Fallback behavior
- [ ] Model forward
- [ ] Model backward
- [ ] Training step
- [ ] Distributed compatibility if applicable

## Performance
- [ ] Warm-up
- [ ] Synchronization
- [ ] Latency baseline
- [ ] Fused latency
- [ ] Memory consumption
- [ ] Result comparison
