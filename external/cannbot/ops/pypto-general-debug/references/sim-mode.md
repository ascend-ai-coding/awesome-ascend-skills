# SIM Mode Limitations (DEBUG_GUIDEBOOK.md §9.6)

*Agent-learned patterns from GDR kernel development.*

## Issue: SIM mode produces garbage values

**Symptom:** Extreme errors (10^20+) in SIM mode, but kernel may work on NPU.

**Root Cause:** SIM mode has fundamental limitations with:
- Dynamic shape handling
- Cube tile configuration
- Memory operations

**Solution:**
1. Verify mathematical correctness against golden reference
2. Test on actual NPU hardware
3. Don't rely on SIM mode for precision validation

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    ...
```

**Warning:** SIM mode is for basic execution testing, NOT precision validation.

---

## Issue: `operand1 dim[0] = -1` in SIM mode

**Symptom:** Dimension becomes -1 when using dynamic shapes in SIM mode.

**Solution:** Use static shapes for SIM mode testing, or accept SIM limitations for dynamic shapes.

---

## Issue: `Invalid tile values: kL0=0, kL1a=0...`

**Symptom:** Cube tiling validation fails in SIM mode.

**Solution:** This is a SIM mode limitation. The kernel may work on actual NPU hardware.

---

## See also

- `npu-launch-failures.md` "SIM mode shows precision failure but NPU works fine" for the same lesson confirmed across delivered ops.
