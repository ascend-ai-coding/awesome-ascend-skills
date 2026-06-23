---
name: remote-npu-test
description: Run NPU inference/training tests on a remote SSH server with vllm-ascend Docker container. Use when the user asks to test models on NPU, run inference on Ascend devices, or deploy models to an SSH server.
when_to_use: User mentions "NPU test", "NPU inference", "Ascend test", "vllm-ascend", "NPU training", or asks to run model tests on a remote server with NPU devices.
argument-hint: "<ssh-host> [image] [devices] [test-command]"
arguments: [ssh-host, image, devices, test-command]
allowed-tools: Bash(ssh *) Bash(docker *) Bash(scp *) Read Grep Glob Edit Write
---

# NPU Inference/Training Test on SSH Server

## Arguments

- `$0` (ssh-host): SSH connection string, e.g., `user@192.168.1.100` or just `hostname` (required)
- `$1` (image): Docker image name. If not provided, auto-detect the highest version `vllm-ascend` image on the remote machine
- `$2` (devices): NPU devices to use, e.g., `0,1,2,3`. If not provided, detect available devices
- `$3` (test-command): The command to run inside the container. If not provided, enter interactive mode

## Instructions

### Step 1: Test SSH Connection

1. Test connectivity to the SSH server:
   ```bash
   ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $0 "echo 'SSH connection successful'"
   ```
2. If connection fails, report the error and ask the user to verify:
   - Host address and port
   - SSH key or password authentication
   - Network connectivity
3. On success, proceed to Step 2.

### Step 2: Detect or Validate Docker Image

**If user specified an image (`$1` is provided):**
- Verify the image exists on the remote server:
  ```bash
  ssh $0 "docker images --format '{{.Repository}}:{{.Tag}}' | grep -w '$1'"
  ```
- If not found, ask the user whether to pull it or select another.

**If no image specified (`$1` is empty):**
- Auto-detect the highest version vllm-ascend image:
  ```bash
  ssh $0 "docker images --format '{{.Repository}}:{{.Tag}}' | grep 'vllm-ascend' | sort -V -t: -k2 | tail -1"
  ```
- If no vllm-ascend image found, report the error and ask the user to provide an image name or pull one.

### Step 3: Detect NPU Devices

**If user specified devices (`$2` is provided):**
- Use the specified devices directly.

**If no devices specified (`$2` is empty):**
- Detect available NPU devices on the remote server:
  ```bash
  ssh $0 "npu-smi info -t board 2>/dev/null || npu-smi info 2>/dev/null"
  ```
- Parse the output to identify available NPU device IDs.
- If detection fails, default to device `0` and inform the user.

### Step 4: Create and Run Docker Container

Create the Docker container with the required device mappings and volume mounts:

```bash
ssh $0 "docker run -d \\
	-it \\ #必须
    --name $NAME \\
    --shm-size=1g \\
    --net=host \\
    --privileged \\
    --device /dev/davinci_manager \\
    --device /dev/hisi_hdc \\
    --device /dev/devmm_svm \\
    $(for d in $(echo $DEVICES | tr ',' ' '); do echo "--device /dev/davinci\$d \\\\"; done) \\
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
    -v /usr/local/dcmi:/usr/local/dcmi \\
    -v /usr/local/sbin:/usr/local/sbin \\
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
    -v /etc/ascend_install.info:/etc/ascend_install.info \\
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \\
    -v /home/:/home \\
    -v /data:/data \\
    -e ASCEND_RT_VISIBLE_DEVICES=$DEVICES \\
    -it $IMAGE bash -c '$TEST_CMD'"
```

Where:
- `$DEVICES` = comma-separated NPU device IDs (e.g., `0,1,2,3`)
- `$IMAGE` = the resolved Docker image name
- `$TEST_CMD` = the user's test command (if provided)
- `$NAME` = the docker container name must end with _claude

**Important device mapping rules:**
- Always include the paramater `-it`
- Always include the three mandatory devices: `/dev/davinci_manager`, `/dev/hisi_hdc`, `/dev/devmm_svm`
- Always include the mandatory volume mounts: `/usr/local/Ascend/driver`, `/usr/local/dcmi`, `/usr/local/sbin`, `/usr/local/bin/npu-smi`, `/etc/ascend_install.info`
- Map each NPU device as `/dev/davinciN` where N is the device ID
- Set `ASCEND_RT_VISIBLE_DEVICES` environment variable to control which NPUs are visible
- Forbidden to do any delete file or dir operation

**If no test command provided:**
- Enter interactive mode with `-it` flag and `bash` as the command.
- Inform the user they are now inside the container.

### Step 5: Handle Dependency Issues

If the test command fails due to missing dependencies:

1. Capture the error output and identify missing packages.
2. Install all missing dependencies inside the container:
   ```bash
   docker exec vllm-ascend-env pip install <missing-package-1> <missing-package-2> ... -i $PIP_MIRRORS
   ```
   Prioritize using the following pip mirror for downloads:
   https://mirrors.huaweicloud.com/repository/pypi/simple
   https://pypi.tuna.tsinghua.edu.cn/simple
   https://mirrors.aliyun.com/pypi/simple/
   only cannot find the package on all three above mirrors then can use the default mirrors.

3. After all dependencies are installed, generate `requirements_ai.txt` with the complete list of newly installed packages:
   ```bash
   docker exec vllm-ascend-env pip freeze > requirements_ai.txt
   ```
4. Copy the requirements file back to the host:
   ```bash
   ssh $0 "docker cp vllm-ascend-env:/requirements_ai.txt /tmp/requirements_ai.txt"
   scp $0:/tmp/requirements_ai.txt ./requirements_ai.txt
   ```
5. Inform the user about the generated `requirements_ai.txt` file.

### Step 6: Error Reporting

If any errors block the script, you should record and fix that. But keep in mind, there is no need to fix all the problems.

1. Create or append to `report.txt` with the following information:
   - Timestamp
   - Error description
   - Full error output/stack trace
   - Attempted solutions
   - Suggested next steps

2. Actively think about solutions:
   - Check if the error is a known issue with vllm-ascend
   - Suggest alternative approaches (different image version, different device configuration)
   - Propose code fixes if the error is in the test script

3. Report findings to the user with actionable recommendations.

### Step 7: Generate Summary Report

After the test completes (success or failure), generate a comprehensive summary report and save it to `report.txt`:

1. **Collect all execution information**:
   - Start time and end time
   - Total execution duration
   - SSH connection details (host, user)
   - Docker image used
   - NPU devices used and their status
   - Container name

2. **Record test results**:
   - Test script path and name
   - List of processed files (if applicable)
   - Inference time for each file
   - Model output scores (color_score, sub_scores, etc.)
   - Any warnings during execution

3. **Document issues and fixes**:
   - List of missing dependencies that were installed
   - Code modifications made for compatibility (e.g., torch.compile fixes)
   - System-level installations (fonts, libraries)
   - Version conflicts resolved

4. **Write the report** in the following format:

```bash
ssh $0 "cat > /home/l00910600/report.txt << 'EOF'
================================================================================
                         NPU Test Execution Report
================================================================================

【基本信息】
执行时间: $START_TIME ~ $END_TIME
总耗时: $DURATION
服务器: $SSH_HOST
Docker 镜像: $IMAGE
NPU 设备: $DEVICES (设备名称: $DEVICE_NAME)
容器名称: $CONTAINER_NAME

【测试脚本】
脚本路径: $TEST_SCRIPT
测试命令: $TEST_CMD

【推理结果】
$(for each processed file):
  文件: $FILENAME
  $OUTPUT(the full output of the target script)

【依赖安装】
$(list all installed packages)

【代码修改】
$(list all code changes made):
  1. $FILE_PATH:$LINE - $DESCRIPTION
  2. ...

【警告信息】
$(list any warnings):
  - $WARNING_1
  - $WARNING_2

【输出文件】
结果保存路径: $OUTPUT_PATH
requirements 文件: $REQUIREMENTS_PATH

================================================================================
                              报告生成时间: $REPORT_TIME
================================================================================
EOF"
```

5. **Also save the report locally** (optional):
   ```bash
   scp $0:/home/l00910600/report.txt ./report_$(date +%Y%m%d_%H%M%S).txt
   ```

6. **Display summary to user**:
   - Print a condensed version of the report in the terminal
   - Highlight key metrics (total time, success/failure, files processed)
   - Mention the full report file location

## Workflow Summary

```
SSH Connection Test
        │
        ▼
   ┌────┴────┐
   │ Connected?│
   └────┬────┘
     Yes│  No → Report & ask user
        ▼
  Detect/Validate Image
        │
        ▼
  Detect NPU Devices
        │
        ▼
  Create Docker Container
        │
        ▼
  Run Test Command
        │
        ▼
  ┌────┴────┐
  │ Success? │
  └────┬────┘
    Yes│  No → Check error type
        │         │
        │    ┌────┴────┐
        │    │Dep issue?│
        │    └────┬────┘
        │      Yes│  No → Record in report.txt
        │         ▼
        │    Install deps
        │    Generate requirements_ai.txt
        ▼
  ┌─────────────┐
  │ Step 7:     │
  │ Generate    │
  │ Summary     │
  │ Report      │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Write to    │
  │ report.txt  │
  │ on server   │
  └──────┬──────┘
         │
         ▼
  Display Results
  to User
```

## Conventions

- Always use `-o ConnectTimeout=10` and `-o StrictHostKeyChecking=no` for SSH commands
- Use `set -e` in shell scripts to fail fast on errors
- Quote all variables to prevent word splitting
- Use `tr ',' ' '` to convert comma-separated device list to space-separated for iteration
- Prefer `pip install` over `pip3 install` inside Docker containers (usually the default)
- Always clean up Docker containers after use (the `--rm` flag handles this)
- Use `scp` to transfer files between local and remote machines
- Append to `report.txt` rather than overwriting to preserve error history

## Examples

### Example 1: Basic NPU test with auto-detected image
```
/npu-test user@192.168.1.100
```
This will SSH to the server, find the highest version vllm-ascend image, detect available NPUs, and enter interactive mode.

### Example 2: Specific image and devices
```
/npu-test user@192.168.1.100 vllm-ascend:v0.7.3 0,1,2,3 "python /data/test_model.py"
```
This will use the specified image, devices 0-3, and run the test script.

### Example 3: Run with all available NPUs
```
/npu-test user@192.168.1.100 "" all "python benchmark.py --model /data/models/llama"
```
This will auto-detect the image, use all available NPUs, and run the benchmark command.
