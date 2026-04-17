# Configuration Reference

Detailed reference for npu-smi configuration commands.

> **Validation note**: This file was updated against `npu-smi set -h` / `clear -h` output captured on a real 910B3 host (`software version 25.5.1`). The previous `temperature`, `power-limit`, `ecc-mode`, `compute-mode`, and `persistence-mode` examples were removed because those set types were not accepted on the validated machine.

## Table of Contents

1. [Discovery](#discovery)
2. [Mode Configuration](#mode-configuration)
3. [Fan Control](#fan-control)
4. [System Settings](#system-settings)
5. [Clear Commands](#clear-commands)
6. [Examples](#examples)

---

## Discovery

### Check Supported Set Types

```bash
npu-smi set -h
npu-smi clear -h
```

The validated host reported these relevant mutable types:

- `ecc-enable`
- `mac-addr`
- `vnpu-mode`
- `create-vnpu` / `destroy-vnpu`
- `p2p-mem-cfg`
- `pwm-mode`
- `pwm-duty-ratio`
- `power-state`
- `boot-select`
- `tls-cert`
- `tls-cert-period`
- `cpu-freq-up`
- `sys-log-enable`

> Always re-run `npu-smi set -h` on the target host before changing configuration, because supported types vary by hardware / firmware.

---

## Mode Configuration

### ECC Enable

```bash
npu-smi set -t ecc-enable -i <id> -c <chip_id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

### P2P Memory Configuration

```bash
npu-smi set -t p2p-mem-cfg -i <id> -c <chip_id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

### vNPU Mode

```bash
npu-smi set -t vnpu-mode -d <value>
```

| Value | Mode |
|-------|------|
| 0 | docker |
| 1 | vm |

---

## Fan Control

### Set Fan Mode

```bash
npu-smi set -t pwm-mode -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Manual |
| 1 | Automatic |

### Set Fan Speed (Manual Mode)

```bash
npu-smi set -t pwm-duty-ratio -d <value>
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| value | 0-100 | Fan speed percentage |

**Note**: Only effective when fan mode is set to Manual (0).

---

## System Settings

### MAC Address

```bash
npu-smi set -t mac-addr -i <id> -c <chip_id> -d <mac_id> -s "XX:XX:XX:XX:XX:XX"
```

| Parameter | Description |
|-----------|-------------|
| mac_id | MAC interface: 0=eth0, 1=eth1, 2=eth2, 3=eth3 |
| mac_string | MAC address format "XX:XX:XX:XX:XX:XX" |

**Note**: Requires restart after change.

### Boot Medium

```bash
npu-smi set -t boot-select -i <id> -c <chip_id> -d <value>
```

| Value | Medium |
|-------|--------|
| 3 | M.2 SSD |
| 4 | eMMC |

**Note**: Requires restart after change.

### CPU Frequency

```bash
npu-smi set -t cpu-freq-up -i <id> -d <value>
```

| Value | CPU Frequency | AI Core Frequency |
|-------|--------------|-------------------|
| 0 | 1.9 GHz | 800 MHz |
| 1 | 1.0 GHz | 800 MHz |

### System Logging

```bash
npu-smi set -t sys-log-enable -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

### Power State

```bash
npu-smi set -t power-state -i <id> -c <chip_id> -d <value>
```

> Check the official documentation and `npu-smi set -h` on the target platform before using `power-state`; supported values are platform-dependent.

---

## Clear Commands

### Clear ECC Errors

```bash
npu-smi clear -t ecc-info -i <id> -c <chip_id>
```

Clears the ECC error counter for the specified device/chip.

### Restore Certificate Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

Restores the certificate expiration threshold to default (90 days).

---

## Examples

### Configure Safe Defaults

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "Applying validated low-risk defaults..."
npu-smi set -t ecc-enable -i $NPU -c $CHIP -d 1
npu-smi set -t p2p-mem-cfg -i $NPU -c $CHIP -d 1
echo "Done!"
```

### Temperature-Based Fan Control

```bash
#!/bin/bash

NPU=0
CHIP=0

# Get current temperature
TEMP=$(npu-smi info -t temp -i $NPU -c $CHIP 2>/dev/null | grep -oP 'NPU Temperature\s*:\s*\K[0-9]+' || echo "0")

# Switch to manual mode
npu-smi set -t pwm-mode -d 0

# Set fan speed based on temperature
if [ $TEMP -gt 70 ]; then
    npu-smi set -t pwm-duty-ratio -d 90
elif [ $TEMP -gt 60 ]; then
    npu-smi set -t pwm-duty-ratio -d 70
else
    npu-smi set -t pwm-duty-ratio -d 50
fi

echo "Temperature: ${TEMP}°C, Fan speed adjusted"
```

### Enable Persistence Mode on All Devices

```bash
#!/bin/bash

mode=$(npu-smi info -t vnpu-mode | grep -oP 'vnpu-mode\s*:\s*\K\w+' || echo "unknown")
echo "Current vNPU mode: $mode"
echo "Switching to docker mode..."
npu-smi set -t vnpu-mode -d 0
echo "Done!"
```

### Enable System Logging

```bash
#!/bin/bash

npu-smi set -t sys-log-enable -d 1
echo "Done!"
```

### Reset All Error Counters

```bash
#!/bin/bash

for npu in $(npu-smi info -l 2>/dev/null | grep -oP 'NPU ID\s*:\s*\K[0-9]+'); do
    for chip in $(npu-smi info -m 2>/dev/null | awk -v npu="$npu" '$1 == npu && $4 ~ /^Ascend/ {print $2}'); do
        echo "Clearing ECC errors on NPU $npu, Chip $chip..."
        npu-smi clear -t ecc-info -i $npu -c $chip
    done
done

echo "Done!"
```
