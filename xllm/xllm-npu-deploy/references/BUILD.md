# XLLM Build Guide

This document describes how to compile xllm framework for NPU deployment.

> **Note**: This is a reference document. The skill will load it on demand when users need compilation help.

## Prerequisites

Before building xllm, ensure you have:

1. **NPU driver and toolkit installed**
   ```bash
   npu-smi info  # Verify NPU is accessible
   ```

2. **Python >= 3.8**

3. **Git submodules initialized**
   ```bash
   git submodule update --init
   ```

## Quick Build

### For Ascend A2 (Atlas 800T A2)

```bash
cd /path/to/xllm

# Build with logging
python3 setup.py build 2>&1 | tee logs/build_a2_$(date +%Y%m%d_%H%M%S).log
```

### For Ascend A3 (Atlas 900 A3)

```bash
cd /path/to/xllm

# Build with --device a3 flag and logging
python3 setup.py build --device a3 2>&1 | tee logs/build_a3_$(date +%Y%m%d_%H%M%S).log
```

## Build Options

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device {a2,a3}` | Target NPU architecture | a2 |
| `--clean` | Clean build (remove build directory first) | False |

### Examples

```bash
# Standard A2 build
python3 setup.py build

# A3 build
python3 setup.py build --device a3

# Clean rebuild for A3
python3 setup.py build --device a3 --clean

# Build with log file
mkdir -p logs
python3 setup.py build --device a3 2>&1 | tee logs/build_$(date +%Y%m%d_%H%M%S).log
```

## Logging Best Practices

### Simple Build with Log

```bash
cd /path/to/xllm
mkdir -p logs

# Build and save log
python3 setup.py build --device a3 2>&1 | tee logs/build.log
```

### Advanced Build Script

```bash
#!/bin/bash
# save as: build_xllm.sh

XLLM_DIR=/path/to/xllm
LOG_DIR=$XLLM_DIR/logs
DEVICE=${1:-a2}  # a2 or a3

mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/build_${DEVICE}_${TIMESTAMP}.log

echo "Building xllm for ${DEVICE}..."
echo "Log file: ${LOG_FILE}"

cd $XLLM_DIR

# Build with logging
python3 setup.py build --device ${DEVICE} 2>&1 | tee ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Binary location: $XLLM_DIR/build/xllm/core/server/xllm"
else
    echo "Build failed! Check ${LOG_FILE}"
    exit 1
fi
```

### Usage

```bash
# Build for A2 (default)
./build_xllm.sh

# Build for A3
./build_xllm.sh a3
```

## Build Output

After successful build, the binary is located at:

```
/path/to/xllm/build/xllm/core/server/xllm
```

Verify the build:

```bash
# Check binary exists
ls -lh build/xllm/core/server/xllm

# Test run
./build/xllm/core/server/xllm --help
```

## First Build Notes

> **Important**: The first compilation takes a long time because all dependencies in vcpkg need to be compiled. Subsequent compilations will be much faster.

## Troubleshooting Build Issues

### Issue: Git submodules not initialized

**Error**: `fatal: not a git repository` or missing dependencies

**Solution**:
```bash
git submodule update --init
```

### Issue: Build fails with NPU errors

**Error**: NPU-related compilation errors

**Solution**: Check device flag
```bash
# For A3, must use --device a3
python3 setup.py build --device a3
```

### Issue: Permission denied

**Error**: `Permission denied` during build

**Solution**:
```bash
# Ensure write permissions
chmod -R u+w .

# Or run with appropriate user
```

### Issue: Out of memory during build

**Error**: Build process killed or OOM errors

**Solution**: The setup.py handles parallel builds internally. If OOM occurs, try:
```bash
# Reduce parallel jobs by setting environment variable
export CMAKE_BUILD_PARALLEL_LEVEL=4
python3 setup.py build --device a3
```

## Clean Build

To perform a clean rebuild:

```bash
cd /path/to/xllm

# Method 1: Use --clean flag
python3 setup.py build --device a3 --clean 2>&1 | tee logs/clean_build.log

# Method 2: Manual clean
rm -rf build
python3 setup.py build --device a3 2>&1 | tee logs/build.log
```

## Release Image Note

If using a release image (with version number in tag), you can skip compilation:

```bash
# Release images have pre-compiled binary at:
/usr/local/bin/xllm
```

## Verify Installation

```bash
# Check binary
ls -lh build/xllm/core/server/xllm

# Run deployment
python /root/.claude/skills/xllm-npu-deploy/scripts/deploy_server.py \
    --model /path/to/model \
    --port 8080 \
    --xllm_binary ./build/xllm/core/server/xllm \
    --daemon \
    --log_file ./logs/server.log
```
