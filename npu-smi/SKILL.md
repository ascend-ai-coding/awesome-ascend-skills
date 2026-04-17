---
name: npu-smi
description: Huawei Ascend NPU npu-smi command reference. Use for device queries (health, temperature, power, memory, processes, ECC), configuration (thresholds, modes, fan), firmware upgrades (MCU, bootloader, VRD), virtualization (vNPU), and certificate management.
keywords:
    - npu-smi
    - 设备管理
    - 温度
    - 功耗
    - 内存
    - 进程
    - ECC
    - 配置
    - 风扇
    - 固件升级
    - 虚拟化
    - 证书管理
---

# npu-smi Command Reference

Quick reference for Huawei Ascend NPU device management commands.

> **Validated on real host**: Ascend 910B3 server (`npu-smi` software version `25.5.1`) on 2026-03-27. Command availability and output fields can differ by platform / firmware.

## Quick Start

```bash
npu-smi info -l                           # List all devices
npu-smi info -t health -i 0               # Check device health
npu-smi info -t temp -i 0 -c 0            # Check temperature
npu-smi info -t power -i 0 -c 0           # Check power
npu-smi info -t memory -i 0 -c 0          # Check memory
```

## Device Queries

### Basic Information

```bash
npu-smi info -l                           # List devices
npu-smi info -t health -i <id>            # Health status (OK/Warning/Error)
npu-smi info -t board -i <id>             # Board details (firmware, software version)
npu-smi info -m                           # List card/chip mapping and chip names
```

### Real-time Metrics

```bash
npu-smi info -t temp -i <id> -c <chip>    # Temperature (NPU, AI Core)
npu-smi info -t power -i <id> -c <chip>   # Power usage and limit
npu-smi info -t memory -i <id> -c <chip>  # Memory usage, total, rate
```

### Advanced Queries

```bash
npu-smi info -t ecc -i <id> -c <chip>     # ECC errors and mode
npu-smi info -t usages -i <id> -c <chip>  # Utilization (AI Core, memory, bandwidth)
npu-smi info -t proc-mem -i <id> -c <chip>   # Per-process memory usage
npu-smi info -t pcie-err -i <id> -c <chip>   # PCIe error counters
npu-smi info -t topo -i <id> -c <chip>       # Inter-NPU topology / affinity
npu-smi info -t p2p-enable -i <id> -c <chip> # P2P capability (if supported)
npu-smi info -t product -i <id> -c <chip> # Product name and serial
```

> **Note**: `npu-smi info -t npu` was **not** accepted on the validated host; use `npu-smi info -m` for chip mapping and `health/temp/power/usages/memory` for per-chip runtime data.

> **See**: [references/device-queries.md](references/device-queries.md) for output formats, examples, monitoring scripts, and **platform identification** (A2 vs A3).

## Configuration

### Temperature and Power Thresholds

```bash
npu-smi set -h                                             # Show supported configurable types on this host
npu-smi set -t pwm-mode -d <0|1>                           # Fan control mode
npu-smi set -t pwm-duty-ratio -d <0-100>                   # Fan duty ratio
```

### Mode Configuration

```bash
npu-smi set -t ecc-enable -i <id> -c <chip> -d <0|1>      # ECC enable switch
npu-smi set -t p2p-mem-cfg -i <id> -c <chip> -d <0|1>     # P2P memory configuration
npu-smi set -t vnpu-mode -d <0|1>                          # vNPU mode
npu-smi set -t cpu-freq-up -i <id> -d <0|1>               # CPU frequency profile
```

### Fan Control

```bash
npu-smi set -t pwm-mode -d <0|1>                          # 0=Manual, 1=Automatic
npu-smi set -t pwm-duty-ratio -d <0-100>                  # Fan speed (percent)
```

### System Settings

```bash
npu-smi set -t mac-addr -i <id> -c <chip> -d <mac_id> -s "XX:XX:XX:XX:XX:XX"
npu-smi set -t boot-select -i <id> -c <chip> -d <3|4>     # 3=M.2 SSD, 4=eMMC
npu-smi set -t cpu-freq-up -i <id> -d <0|1>               # 0=1.9GHz/800MHz, 1=1.0GHz/800MHz
npu-smi set -t sys-log-enable -d <0|1>                    # System logging
```

### Clear Commands

```bash
npu-smi clear -t ecc-info -i <id> -c <chip>               # Clear ECC errors
npu-smi clear -t tls-cert-period -i <id> -c <chip>        # Restore cert threshold
```

> **See**: [references/configuration.md](references/configuration.md) for parameter tables and examples.

## Firmware Management

### Upgrade Workflow

```
Query → Upgrade → Check Status → Activate → Restart
```

```bash
npu-smi upgrade -b <item> -i <id>                         # Query current version
npu-smi upgrade -t <item> -i <id> -f <file.hpm>           # Upload firmware
npu-smi upgrade -q <item> -i <id>                         # Check upgrade status
npu-smi upgrade -a <item> -i <id>                         # Activate firmware
```

### Components and Restart Requirements

| Component | Item Name | Restart Required |
|-----------|-----------|------------------|
| MCU | `mcu` | Yes (restart) |
| Bootloader | `bootloader` | Yes (restart) |
| VRD | `vrd` | Yes (power cycle 30s) |

> **See**: [references/firmware-upgrade.md](references/firmware-upgrade.md) for complete procedures.

## Virtualization (vNPU)

### Queries

```bash
npu-smi info -t vnpu-mode                                 # Query AVI mode (0=Container, 1=VM)
npu-smi info -t template-info                             # List all templates
npu-smi info -t template-info -i <id>                     # Templates for specific device
npu-smi info -t info-vnpu -i <id> -c <chip>               # View vNPU info
```

### Management

```bash
npu-smi set -t vnpu-mode -d <0|1>                         # Set AVI mode
npu-smi set -t create-vnpu -i <id> -c <chip> -f <template> [-v <vnpu_id>] [-g <vgroup_id>]
npu-smi set -t destroy-vnpu -i <id> -c <chip> -v <vnpu_id>
```

**vNPU ID Range**: `[phy_id*16+100, phy_id*16+115]`

> **See**: [references/virtualization.md](references/virtualization.md) for vNPU creation and management.

## Certificate Management

### Queries

```bash
npu-smi info -t tls-csr-get -i <id> -c <chip>             # Generate CSR (PEM format)
npu-smi info -t tls-cert -i <id> -c <chip>                # View certificate details
npu-smi info -t tls-cert-period -i <id> -c <chip>         # Check expiration threshold
npu-smi info -t rootkey -i <id> -c <chip>                 # Rootkey status
```

### Management

```bash
npu-smi set -t tls-cert -i <id> -c <chip> -f "<tls.pem> <ca.pem> <subca.pem>"
npu-smi set -t tls-cert-period -i <id> -c <chip> -s <days>  # Set threshold (7-180 days)
npu-smi clear -t tls-cert-period -i <id> -c <chip>        # Restore default (90 days)
```

> **See**: [references/certificate-management.md](references/certificate-management.md) for certificate lifecycle management.

## Parameters Reference

| Parameter | Description | How to Get |
|-----------|-------------|------------|
| `id` | Device ID (NPU ID) | `npu-smi info -l` |
| `chip_id` | Runtime chip ID | `npu-smi info -m` (usually `0` for Ascend chip, `1` for MCU on validated host) |
| `vnpu_id` | vNPU ID | Auto-assigned or specified in range |
| `mac_id` | MAC interface | 0=eth0, 1=eth1, 2=eth2, 3=eth3 |

## Supported Platforms

- Atlas 200I DK A2 Developer Kit
- Atlas 500 A2 Smart Station
- Atlas 200I A2 Acceleration Module (RC/EP scenarios)
- Atlas A2/A3 Training Series
- Atlas Training Series

> **Note**: Chip name (e.g., 910B3) does **not** indicate server platform (A2 vs A3). Use `dmidecode -t system | grep Product` or `npu-smi info -t product` to identify the server model. See [references/device-queries.md](references/device-queries.md#platform-identification) for details.

## Important Notes

- Most configuration commands require **root permissions**
- Device IDs from `npu-smi info -l`
- Chip IDs from `npu-smi info -m`
- MCU/bootloader upgrades require **restart** after activation
- VRD upgrades require **power cycle** (30+ seconds off)
- MAC/boot changes require **restart**
- Command availability varies by hardware platform
- `npu-smi info proc` was **not supported** on the validated 910B3 host; `npu-smi info -t proc-mem` worked
- `npu-smi info -t product` may return `This device does not support querying product.` on some server SKUs

## Scripts

- [scripts/npu-health-check.sh](scripts/npu-health-check.sh) - Comprehensive device health check

## Official Documentation

- **npu-smi Reference**: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html
