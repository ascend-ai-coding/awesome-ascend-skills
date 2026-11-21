# Calibrate an A5 roofline for the measured access pattern

Use this model only for a confirmed DAV_3510 / dav-c310 target. Resolve the exact SKU and active core quota; configurations such as 950PR and 950DT are not interchangeable. The architecture number alone does not identify the SKU. Use the platform configuration matching the installed CANN and PyPTO revisions, and retain its resolved path and revision or file hash.

The [versioned platform configuration directory](https://gitcode.com/cann/pypto/tree/dbf64c4aae4e75fcdc27c4d077054b9fafbb1462/framework/src/platform/parser/simulation_platform/platform_config) is a source-location example. Resolve the matching file in the installed source tree and check `[version] SoC_version`; discover the installed layout if it differs. A file from another installation does not establish the current numerical ceiling.

Read the matching operand dtype's hardware tuple from `DtypeMKN`; its M/K/N values are not the workload's matrix dimensions. After verifying that this tuple describes work per cycle, count a multiply-add as two FLOPs and convert `cube_freq` to Hz before deriving the nominal rate:

```text
cube_ops_per_cycle   = dtype_M * dtype_K * dtype_N * 2
cube_peak_flops_per_s = cube_ops_per_cycle * active_cube_cores * cube_clock_hz
```

This Cube rate does not model Vector work. Verify the units of `ddr_rate` and directional transfer rates before treating them as bandwidth; adding Cube and Vector rates also requires evidence that both engines can sustain those rates concurrently. Use the existing [bandwidth definitions](performance.md#distinguish-bandwidth-quantities) to choose a matching sustained baseline and distinguish it from kernel effective throughput. Leave an unresolved ceiling unquantified.
