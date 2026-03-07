# op-plugin reference

## Version tables (from op-plugin README)

**op-plugin branch ↔ Ascend Extension for PyTorch (torch_npu):**

| op-plugin branch | torch_npu version (branch)   |
|------------------|-----------------------------|
| master           | mainline, e.g. v2.7.1       |
| 7.3.0            | e.g. v2.7.1-7.3.0          |
| 7.2.0            | e.g. v2.7.1-7.2.0          |
| 7.1.0 / 7.0.0 / 6.x / 5.x | see op-plugin README |

**Build script:** `bash ci/build.sh --python=<ver> --pytorch=<branch>`. Example: `--python=3.9 --pytorch=v2.7.1-7.3.0`.

**PyTorch / Python / GCC (representative):**

| PyTorch   | Python        | GCC (ARM / x86) |
|-----------|---------------|------------------|
| v2.6.0    | 3.9, 3.10, 3.11 | 11.2 / 9.3       |
| v2.7.1    | 3.9, 3.10, 3.11 | 11.2             |
| v2.8.0+   | 3.9, 3.10, 3.11 (3.10+ for 2.9+) | 13.3   |

PyTorch 2.6+ does not support Python 3.8. See the op-plugin README for the exact matrix.

## SOC_VERSION

- In CMakeLists.txt: `set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")`.
- Get chip name on the machine: run `npu-smi info` and read **Chip Name**. Set SOC_VERSION to `Ascend` + that value (e.g. Ascend910B).

## Links

- [op-plugin (gitcode)](https://gitcode.com/ascend/op-plugin)
- [Ascend Extension for PyTorch (torch_npu)](https://gitcode.com/ascend/pytorch)
- op-plugin repo `examples/cpp_extension/README.md` — directory layout, kernel/host/tiling, run steps (path under repo: examples/cpp_extension/README.md)
- [Ascend C](https://www.hiascend.com/ascend-c) — Ascend C kernel development

