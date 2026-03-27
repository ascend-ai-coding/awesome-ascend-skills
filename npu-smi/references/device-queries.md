# Device Queries Reference

Detailed reference for npu-smi device query commands.

> **Validation note**: The examples below were corrected against a real 910B3 host (`npu-smi` software version `25.5.1`). Older output snippets based on `NPU : 0` / `Name : 910B3` were removed because current hosts expose different fields.

## Table of Contents

1. [Platform Identification](#platform-identification)
2. [Basic Queries](#basic-queries)
3. [Real-time Metrics](#real-time-metrics)
4. [Advanced Queries](#advanced-queries)
5. [Output Formats](#output-formats)
6. [Monitoring Scripts](#monitoring-scripts)

---

## Platform Identification

> **Important**: Chip name alone does **NOT** determine the server platform (A2 vs A3).

### Common Misconception

When `npu-smi info -m` shows **Chip Name: Ascend 910B3**, this does **NOT** mean the machine is an **Atlas A3**. The same chip (910B3) can be used in both **A2** and **A3** servers.

**Example**: An Atlas A2 server with 8× 910B3 chips will still show "910B3" as the chip name.

### Correct Method: Check System Product Info

To identify whether you have an **Atlas A2** or **Atlas A3** server, use system-level commands:

```bash
# Method 1: Using dmidecode (recommended)
dmidecode -t system 2>/dev/null | head -20 | grep Product

# Method 2: Check product information via npu-smi
npu-smi info -t product -i 0 -c 0
```

### Platform Mapping Reference

| Chip Name | Server Platform |
|-----------|-----------------|
| Ascend 910B | Atlas A2 |
| Ascend 910C | Atlas A3 |
| Ascend 950 | Atlas A5 |

### Key Takeaway

- **Chip name** indicates the NPU processor model
- **Server platform** (A2/A3) is determined by the system product information
- Always verify via `dmidecode` or `npu-smi info -t product` rather than assuming from chip name

---

## Basic Queries

### List Devices

```bash
npu-smi info -l
```

**Output Fields**:
| Field | Description |
|-------|-------------|
| Total Count | Number of visible NPUs |
| NPU ID | Device identifier |
| Chip Count | Number of chips exposed under that card |

**Example Output**:
```
Total Count                    : 8

NPU ID                         : 0
Chip Count                     : 1

NPU ID                         : 1
Chip Count                     : 1
```

### Query Device Health

```bash
npu-smi info -t health -i <id>
```

**Output**:
| Field | Values |
|-------|--------|
| Healthy | OK, Warning, Error |

### Query Board Information

```bash
npu-smi info -t board -i <id>
```

**Output Fields**:
| Field | Description |
|-------|-------------|
| NPU ID | Device identifier |
| Name | Board name |
| Health | Health status |
| Power Usage | Current power draw |
| Temperature | Board temperature |
| Firmware Version | Current firmware |
| Software Version | Driver version |

### List Card / Chip Mapping

```bash
npu-smi info -m
```

**Output Fields**:
| Field | Description |
|-------|-------------|
| NPU ID | Parent device |
| Chip ID | Chip identifier |
| Chip Logic ID | Logical device id used by runtime |
| Chip Name | Chip name (for example `Ascend 910B3`, `Mcu`) |

**Example Output**:
```
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend 910B3
0                              1                              -                              Mcu
```

> `npu-smi info -t npu` was rejected by the validated host with `Error parameter of -t`; use `info -m` plus specific metric queries instead.

### List All Chips

```bash
npu-smi info -m
```

**Output Fields**:
| Field | Description |
|-------|-------------|
| NPU ID | Parent device |
| Chip ID | Chip identifier |
| Name | Chip name |
| Health | Health status |

---

## Real-time Metrics

### Temperature

```bash
npu-smi info -t temp -i <id> -c <chip_id>
```

**Output**:
- NPU Temperature (°C)
- AI Core Temperature (°C)

**Note**: Output format may vary by npu-smi version.

### Power

```bash
npu-smi info -t power -i <id> -c <chip_id>
```

**Output**:
- Power Usage (W)
- Power Limit (W)

### Memory

```bash
npu-smi info -t memory -i <id> -c <chip_id>
```

**Output**:
- DDR Capacity / Clock (platform-dependent, can be `0`)
- HBM Capacity / Clock
- HBM Temperature
- HBM Manufacturer ID

**Validated Example**:
```
DDR Capacity(MB)               : 0
DDR Clock Speed(MHz)           : 0
HBM Capacity(MB)               : 65536
HBM Clock Speed(MHz)           : 1600
HBM Temperature(C)             : 52
HBM Manufacturer ID            : 0x56
```

> On the validated host, usage percentage is exposed by `npu-smi info -t usages`, not `npu-smi info -t memory`.

---

## Advanced Queries

### Running Processes

```bash
npu-smi info -t proc-mem -i <id> -c <chip_id>
```

**Note**: `npu-smi info proc -i <id> -c <chip_id>` was **not supported** on the validated host, but `-t proc-mem` worked.

**Output Fields**:
| Field | Description |
|-------|-------------|
| PID | Process ID |
| Process Name | Application name |
| Memory Usage | Memory used |

**Validated Example**:
```
Process id:207795  Process name:VLLMEngineCor     Process memory(MB):55436
```

### ECC Errors

```bash
npu-smi info -t ecc -i <id> -c <chip_id>
```

**Output**:
- ECC Error Count
- ECC Mode (Enabled/Disabled)

### Utilization

```bash
npu-smi info -t usages -i <id> -c <chip_id>
```

**Output**:
- AI Core Usage (%)
- Memory Usage (%)
- Bandwidth Usage (%)

### PCIe Errors

```bash
npu-smi info -t pcie-err -i <id> -c <chip_id>
```

**Output**:
- TX Error Count
- RX Error Count
- LCRC Error Count
- ECRC Error Count
- Retry Count

### Topology

```bash
npu-smi info -t topo -i <id> -c <chip_id>
```

**Output**:
- NPU-to-NPU connectivity matrix
- CPU affinity mapping

### P2P Capability

```bash
npu-smi info -t p2p-enable -i <id> -c <chip_id>
```

**Note**: Some devices return `This device does not support querying p2p-enable.`

### Product Info

```bash
npu-smi info -t product -i <id> -c <chip_id>
```

**Output**:
- Product Name
- Product Serial Number

**Note**: Some server products return `This device does not support querying product.` even though the command exists.

---

## Output Formats

Output format may vary by:
- npu-smi version
- Hardware platform
- Firmware version

Always verify output format on your specific system.

---

## Monitoring Scripts

### Quick Health Check

```bash
#!/bin/bash

# Check health of all devices
npu-smi info -l | grep -oP 'NPU ID\s*:\s*\K[0-9]+' | while read -r npu; do
    health=$(npu-smi info -t health -i "$npu" | grep -oP 'Health\s*:\s*\K\w+' | head -n1)
    echo "NPU $npu: $health"
done
```

### Device Summary

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Device $NPU Summary ==="
npu-smi info -t health -i $NPU
npu-smi info -t board -i $NPU
echo ""
echo "=== Mapping ==="
npu-smi info -m
echo ""
echo "=== Metrics ==="
npu-smi info -t temp -i $NPU -c $CHIP
npu-smi info -t power -i $NPU -c $CHIP
npu-smi info -t memory -i $NPU -c $CHIP
npu-smi info -t usages -i $NPU -c $CHIP
```

### Resource Monitoring Script

```bash
#!/bin/bash

echo "=== NPU Resource Monitor $(date) ==="

for npu in $(npu-smi info -l 2>/dev/null | grep -oP 'NPU ID\s*:\s*\K[0-9]+'); do
    echo ""
    echo "--- NPU $npu ---"
    temp=$(npu-smi info -t temp -i $npu -c 0 2>/dev/null | grep -oP 'NPU Temperature \(C\)\s*:\s*\K[0-9]+' || echo "N/A")
    power=$(npu-smi info -t power -i $npu -c 0 2>/dev/null | grep -oP 'NPU Real-time Power\(W\)\s*:\s*\K[0-9.]+' || echo "N/A")
    mem=$(npu-smi info -t usages -i $npu -c 0 2>/dev/null | grep -oP 'HBM Usage Rate\(%\)\s*:\s*\K[0-9]+' || echo "N/A")
    echo "  Temperature: ${temp}°C"
    echo "  Power: ${power}W"
    echo "  HBM Usage: ${mem}%"
done
```

### Error Check Script

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Error Check ==="
echo "ECC Errors:"
npu-smi info -t ecc -i $NPU -c $CHIP

echo ""
echo "Health Status:"
npu-smi info -t health -i $NPU
```

### Full System Report

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Advanced System Report ==="
echo ""
echo "Process memory:"
npu-smi info -t proc-mem -i $NPU -c $CHIP 2>/dev/null || echo "Process memory info not available"

echo ""
echo "ECC Status:"
npu-smi info -t ecc -i $NPU -c $CHIP

echo ""
echo "Utilization:"
npu-smi info -t usages -i $NPU -c $CHIP

echo ""
echo "PCIe Errors:"
npu-smi info -t pcie-err -i $NPU -c $CHIP

echo ""
echo "Topology:"
npu-smi info -t topo -i $NPU -c $CHIP
```
