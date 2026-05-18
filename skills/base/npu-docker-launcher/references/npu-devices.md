# NPU Devices Reference

## Overview

Ascend NPU (Neural Processing Unit) devices are used for AI acceleration on Huawei Ascend platforms. When launching Docker containers that need to access NPU cards, proper device mounting and configuration is required.

## Querying Available NPU Cards

### Using `npu-smi info`

The primary command to query available NPU cards:

```bash
npu-smi info
```

Example output:
```
+----------------------------------------------------------------+
| npu-smi 23.0.rc3.0                                             |
+-----------------+--------------+-------------------+-----------+
| NPU     Name     | Health      | Power(W)          | Temp(C)   |
| Index   |         |             |                   |           |
+=================+==============+===================+===========+
| 0       | 910B     | OK          | 70.2              | 52        |
+-----------------+--------------+-------------------+-----------+
| 1       | 910B     | OK          | 71.5              | 53        |
+-----------------+--------------+-------------------+-----------+
```

### Alternative Commands

```bash
# List NPU devices
ls -l /dev/davinci*

# Check NPU driver version
cat /usr/local/Ascend/driver/version.info

# Check for installed Ascend packages
pip list | grep ascend
```

## NPU Device Naming Convention

NPU devices are exposed through the `/dev/davinci*` character devices:

- `/dev/davinci0` - First NPU card (index 0)
- `/dev/davinci1` - Second NPU card (index 1)
- `/dev/davinci2` - Third NPU card (index 2)
- `/dev/davinci3` - Fourth NPU card (index 3)
- ... and so on

The number after `davinci` corresponds to the NPU card index shown in `npu-smi info`.

## Device Mounting Strategies

### 1. Privileged Mode (Simplest)

Grants the container full access to all host devices:

```bash
docker run --privileged ...
```

**Pros**: Simple, no need to specify individual devices
**Cons**: Less secure, grants more permissions than needed

### 2. Specific Device Mounting (More Secure)

Mount only the required NPU devices:

```bash
docker run --device=/dev/davinci0 --device=/dev/davinci1 ...
```

**Pros**: More secure, principle of least privilege
**Cons**: Need to know which devices are available and required

## Common Ascend Driver Paths

When mounting Ascend driver resources, these directories are commonly needed:

### Required Directories

```bash
/usr/local/Ascend/driver      # Main driver directory
/usr/local/sbin                # System binaries (may be required)
```

### Optional Directories (depending on use case)

```bash
/usr/local/Ascend/ascend-toolkit  # Ascend toolkit if installed
/usr/local/Ascend/nnae           # NNAE (Neural Network Acceleration Engine)
/usr/local/Ascend/opp            # Operator development kit
```

## Docker Mount Examples

### Single NPU Card (Card 0)

```bash
docker run \
  --device=/dev/davinci0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  ...
```

### Multiple NPU Cards (Cards 0-3)

```bash
docker run \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  ...
```

### Privileged Mode (All Devices)

```bash
docker run \
  --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  ...
```

## Automatic Device Detection

To automatically detect available NPU cards and generate device mounts:

```bash
# Get the number of NPU cards
NPU_COUNT=$(ls /dev/davinci* 2>/dev/null | wc -l)

# Generate device arguments
DEVICE_ARGS=""
for i in $(seq 0 $((NPU_COUNT - 1))); do
  DEVICE_ARGS="$DEVICE_ARGS --device=/dev/davinci$i"
done

echo "Detected $NPU_COUNT NPU card(s)"
echo "Device arguments: $DEVICE_ARGS"
```

## Troubleshooting

### Permission Denied

If you get permission denied accessing `/dev/davinci*`:

```bash
# Check device permissions
ls -l /dev/davinci*

# Add user to required group (typically 'root' or 'huawei')
sudo usermod -aG huawei $USER
```

### Device Not Found

If `/dev/davinci*` devices don't exist:

1. Check if Ascend driver is installed:
   ```bash
   cat /usr/local/Ascend/driver/version.info
   ```

2. Check if NPU hardware is detected:
   ```bash
   lspci | grep Huawei
   ```

3. Restart the driver service:
   ```bash
   sudo systemctl restart ascend-driver
   ```

### Container Can't Access NPU

1. Verify device mounting: `docker inspect <container> | grep -A 10 Devices`
2. Check if using privileged mode if needed
3. Verify driver directories are mounted
4. Check environment variables for Ascend configuration:
   ```bash
   docker exec <container> env | grep ASCEND
   ```

## Environment Variables

Common Ascend-related environment variables:

```bash
ASCEND_HOME=/usr/local/Ascend
ASCEND_DRIVER_DIR=/usr/local/Ascend/driver
ASCEND_DEVICE_ID=0           # Which NPU card to use
ASCEND_SLOG_PRINT_TO_STDOUT=1  # Enable logging to stdout
ASCEND_GLOBAL_LOG_LEVEL=1    # Log level (0=DEBUG, 1=INFO, etc.)
```

## References

- [Ascend Docker Container Documentation](https://www.hiascend.com/document)
- [npu-smi Command Reference](https://www.hiascend.com/document)
