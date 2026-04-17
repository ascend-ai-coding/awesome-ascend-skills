# Certificate Management Reference

Detailed reference for npu-smi certificate management commands.

## Table of Contents

1. [Overview](#overview)
2. [Certificate Queries](#certificate-queries)
3. [Certificate Management](#certificate-management)
4. [Monitoring](#monitoring)
5. [Examples](#examples)

---

## Overview

npu-smi supports TLS certificate management for secure device communication:
- Generate Certificate Signing Requests (CSR)
- Import signed certificates
- Monitor certificate expiration
- Configure expiration thresholds

---

## Certificate Queries

### Generate CSR

```bash
npu-smi info -t tls-csr-get -i <id> -c <chip_id>
```

**Interactive behavior**: the command first prompts for `country|province|city|organization|department`, then generates the CSR.

**Validated non-interactive example**:

```bash
printf 'CN|ZHEJIANG|HANGZHOU|Huawei|Lab\n' | \
  npu-smi info -t tls-csr-get -i 0 -c 0
```

**Validated output**:

```text
Message                        : The tls csr file of the chip is obtained successfully.
```

**Usage**:
1. Generate CSR using npu-smi
2. Submit CSR to your Certificate Authority (CA)
3. Receive signed certificate from CA
4. Import certificate using `npu-smi set -t tls-cert`

### View Certificate

```bash
npu-smi info -t tls-cert -i <id> -c <chip_id>
```

**Output Fields**:
| Field | Description |
|-------|-------------|
| Subject | Certificate subject |
| Issuer | Certificate issuer (CA) |
| Valid From | Certificate validity start date |
| Valid Until | Certificate expiration date |
| Serial Number | Certificate serial number |

### Query Expiration Threshold

```bash
npu-smi info -t tls-cert-period -i <id> -c <chip_id>
```

**Output**: Number of days before expiration warning

### Query Rootkey Status

```bash
npu-smi info -t rootkey -i <id> -c <chip_id>
```

**Output**: Rootkey status and information

---

## Certificate Management

### Import TLS Certificate

```bash
npu-smi set -t tls-cert -i <id> -c <chip_id> -f "<tls.pem> <ca.pem> <subca.pem>"
```

**Parameters**:
| Parameter | Description |
|-----------|-------------|
| tls.pem | Device certificate (signed by CA) |
| ca.pem | Root CA certificate |
| subca.pem | Intermediate CA certificate (if applicable) |

**Note**: All certificates must be in PEM format.

### Set Expiration Threshold

```bash
npu-smi set -t tls-cert-period -i <id> -c <chip_id> -s <days>
```

**Parameters**:
| Parameter | Range | Default |
|-----------|-------|---------|
| days | 7-180 | 90 |

**Purpose**: Warning threshold for certificate expiration. When certificate is within this many days of expiration, alerts should be triggered.

### Restore Default Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

Restores the expiration threshold to default value (90 days).

---

## Monitoring

### Manual Expiration Check

```bash
#!/bin/bash

NPU=0
CHIP=0

# Get expiration date
EXPIRY=$(npu-smi info -t tls-cert -i $NPU -c $CHIP | grep "Valid Until" | cut -d: -f2-)
echo "Certificate expires: $EXPIRY"

# Get threshold
THRESHOLD=$(npu-smi info -t tls-cert-period -i $NPU -c $CHIP | grep -o '[0-9]*')
echo "Warning threshold: $THRESHOLD days"
```

### Expiration Alert Script

```bash
#!/bin/bash

THRESHOLD=30

echo "=== Certificate Expiration Check ==="

npus=$(npu-smi info -l 2>/dev/null | grep -oP 'NPU ID\s*:\s*\K[0-9]+' || echo "")

if [ -z "$npus" ]; then
    echo "ERROR: No NPU devices found or npu-smi not available"
    exit 1
fi

for npu in $npus; do
    chips=$(npu-smi info -m 2>/dev/null | awk -v npu="$npu" '$1 == npu && $4 ~ /^Ascend/ {print $2}')
    if [ -z "$chips" ]; then
        echo "INFO: NPU $npu - No chips found"
        continue
    fi
    
    for chip in $chips; do
        cert_info=$(npu-smi info -t tls-cert -i $npu -c $chip 2>/dev/null)
        
        # Check if certificate query failed or no certificate exists
        if [ $? -ne 0 ]; then
            echo "INFO: NPU $npu, Chip $chip - Certificate query failed or not supported"
            continue
        fi
        
        # Check for "No certificate" or empty response
        if echo "$cert_info" | grep -qi "no certificate\|not found\|none"; then
            echo "INFO: NPU $npu, Chip $chip - No certificate installed"
            continue
        fi
        
        expiry=$(echo "$cert_info" | grep -i "Valid Until" | cut -d: -f2- | xargs)
        
        if [ -z "$expiry" ]; then
            echo "INFO: NPU $npu, Chip $chip - Could not parse certificate expiration"
            continue
        fi
        
        # Parse expiration date (handle different date formats)
        expiry_ts=$(date -d "$expiry" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$expiry" +%s 2>/dev/null || echo "0")
        
        if [ "$expiry_ts" = "0" ]; then
            echo "WARN: NPU $npu, Chip $chip - Could not parse expiration date: $expiry"
            continue
        fi
        
        now_ts=$(date +%s)
        days_left=$(( (expiry_ts - now_ts) / 86400 ))
        
        if [ $days_left -lt 0 ]; then
            echo "CRITICAL: NPU $npu, Chip $chip - Certificate EXPIRED $((-days_left)) days ago"
        elif [ $days_left -lt $THRESHOLD ]; then
            echo "WARNING: NPU $npu, Chip $chip expires in $days_left days"
        else
            echo "OK: NPU $npu, Chip $chip - $days_left days remaining"
        fi
    done
done
```

---

## Examples

### Check Certificate Status

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "Certificate Info:"
npu-smi info -t tls-cert -i $NPU -c $CHIP

echo ""
echo "Expires in threshold:"
npu-smi info -t tls-cert-period -i $NPU -c $CHIP
```

### Full Certificate Import Process

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Certificate Import Process ==="

# Step 1: Generate CSR (interactive subject string)
echo "1. Generating CSR..."
printf 'CN|ZHEJIANG|HANGZHOU|Huawei|Lab\n' | \
  npu-smi info -t tls-csr-get -i $NPU -c $CHIP > device.csr
echo "CSR request written to device.csr"
echo ""

# Step 2: Submit CSR to CA (manual step)
echo "2. Submit device.csr to your Certificate Authority"
echo "   Receive: device.pem, ca-root.pem, ca-sub.pem"
echo ""
read -p "Press Enter when you have the certificates..."

# Step 3: Import certificates
echo "3. Importing certificates..."
npu-smi set -t tls-cert -i $NPU -c $CHIP \
  -f "device.pem ca-root.pem ca-sub.pem"

# Step 4: Set threshold
echo "4. Setting 60-day threshold..."
npu-smi set -t tls-cert-period -i $NPU -c $CHIP -s 60

# Step 5: Verify
echo "5. Verifying..."
npu-smi info -t tls-cert -i $NPU -c $CHIP

echo ""
echo "Done!"
```

### Set Up Monitoring with Cron

```bash
# Add to crontab (daily check at 8:00 AM)
# 0 8 * * * /path/to/cert-check.sh | mail -s "NPU Certificate Check" admin@example.com
```

### Certificate Rotation

```bash
#!/bin/bash

NPU=0
CHIP=0

echo "=== Certificate Rotation ==="

# Step 1: Generate new CSR
echo "1. Generating new CSR..."
printf 'CN|ZHEJIANG|HANGZHOU|Huawei|Lab\n' | \
  npu-smi info -t tls-csr-get -i $NPU -c $CHIP > device_new.csr

# Step 2: Get new certificate from CA (manual step)
echo "2. Submit device_new.csr to CA and get new certificates"
echo ""
read -p "Press Enter when ready to import new certificates..."

# Step 3: Import new certificates
echo "3. Importing new certificates..."
npu-smi set -t tls-cert -i $NPU -c $CHIP \
  -f "device_new.pem ca-root.pem ca-sub.pem"

# Step 4: Verify
echo "4. Verifying new certificate..."
npu-smi info -t tls-cert -i $NPU -c $CHIP

echo ""
echo "Certificate rotation complete!"
```

## Best Practices

1. **Regular Monitoring**: Check certificate expiration weekly
2. **Proactive Renewal**: Renew certificates before expiration threshold
3. **Backup Certificates**: Keep copies of all certificates and CSRs
4. **Document Process**: Maintain records of certificate issuance and rotation
5. **Automated Alerts**: Set up monitoring for expiration warnings
