---
name: npu-docker-launcher
description: "Directly launch Docker containers on Ascend NPU servers for users. Collect requirements and execute docker run commands with proper NPU device mounting, networking, volume mounts, and environment variables. Use when users need to: (1) Start a Docker container on an NPU server, (2) Create containers with NPU device access, (3) Configure and launch containers with specific networking or mount requirements."
---

# NPU Docker Container Launcher

## Overview

This skill directly launches Docker containers on Ascend NPU servers. When invoked, perform pre-launch checks, gather user requirements, and execute the appropriate `docker run` command.

## Workflow

### 1. Pre-Launch Environment Checks

**MUST perform these checks before proceeding:**

#### 1.1 Check NPU Driver Availability
```bash
# Check if Ascend driver is installed
ls -l /usr/local/Ascend/driver/

# Check driver version (optional)
cat /usr/local/Ascend/driver/version.info 2>/dev/null || echo "Driver version info not found"
```

If driver directory doesn't exist, **STOP** and inform user: "Ascend driver not found at /usr/local/Ascend/driver/. Please install the driver first."

#### 1.2 Check NPU Device Availability
```bash
# List NPU devices
ls -l /dev/davinci* 2>/dev/null

# Check NPU status using npu-smi
npu-smi info 2>/dev/null || echo "npu-smi not available"
```

If no `/dev/davinci*` devices found, **STOP** and inform user: "No NPU devices detected. Please check hardware and driver installation."

#### 1.3 Check Docker Service
```bash
# Check if Docker is running
docker info > /dev/null 2>&1 && echo "Docker is running" || echo "Docker is not running"
```

If Docker is not running, **STOP** and inform user.

### 2. Query Available Docker Images

**Query and display available images before asking user:**

```bash
# List all local Docker images
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
```

Display the images to the user in a formatted table. Then ask about their task and recommend appropriate images based on:
- **PyTorch/TensorFlow training**: Look for images with "pytorch", "tensorflow", "ascend-pytorch", "ascend-tf"
- **MindSpore training**: Look for images with "mindspore"
- **Inference**: Look for images with "inference", "serving", "vllm", "mindie"
- **Development**: Look for images with "dev", "devel", or general framework images

### 3. Gather Requirements

After showing available images and getting user's task, ask:

```
## Docker Container Configuration

1. Container name: (e.g., my-pytorch-env)
2. Privileged mode: (default: yes) - Grants full device access [yes/no]
3. Network mode: (default: host) or specify port mappings [host/custom]
4. NPU cards: Which cards to use? (default: all, or specific like "0,1,2") [all/specific]
5. Mount directories: (format: host:container, comma-separated)
   Example: /home/user:/home/user,/data:/data
6. Work directory inside container: (e.g., /home/user)
7. Environment variables: (space-separated, e.g., "VAR1=value1 VAR2=value2")
8. Additional options: (e.g., auto-remove --rm, shm-size)
```

### 4. Build Docker Run Command

**Default command structure:**
```bash
docker run -it -d \
  --name <container-name> \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v <user-mounts> \
  -w <work-dir> \
  [-e ENV_VARS...] \
  <image> \
  /bin/bash
```

**Key defaults:**
- `-it -d`: Interactive terminal + detached mode (ALWAYS use both)
- `/bin/bash`: Default command to run (ALWAYS include at the end)
- `--privileged`: Default for NPU access
- `--network host`: Default network mode

#### Command Components:

**Required Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `-it -d` | Always | Interactive + detached |
| `--name` | Required | Container name |
| `--privileged` | yes | Full device access for NPU |
| `--network host` | host | Network mode |

**Driver Mounts (ALWAYS include):**
```bash
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/sbin:/usr/local/sbin
```

**Optional Arguments:**
| Argument | When to Use |
|----------|-------------|
| `--device=/dev/davinci0` | Specific NPU card (instead of --privileged) |
| `-p host:container` | Port mapping (instead of --network host) |
| `--shm-size=16g` | Large shared memory for distributed training |
| `--rm` | Auto-remove container on exit |
| `-e VAR=value` | Environment variables |

### 5. Show Command and Confirm

Display the complete command and ask for confirmation:

```
📋 Generated Docker Run Command:

docker run -it -d \
  --name my-container \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /home/user:/home/user \
  -w /home/user \
  pytorch/pytorch:latest \
  /bin/bash

Proceed with container launch? [yes/no]
```

### 6. Execute and Verify

After user confirms:

1. **Execute the command** using Bash tool
2. **Verify container is running**:
   ```bash
   docker ps | grep <container-name>
   ```
3. **Report status** with access instructions

### 7. Report Status

```
✅ Container started successfully
- Container ID: abc123def456
- Container name: my-container
- Image: pytorch/pytorch:latest
- Status: Running

📌 Access Instructions:
- Enter container: docker exec -it my-container bash
- View logs: docker logs -f my-container
- Stop container: docker stop my-container
- Remove container: docker rm my-container
```

## Decision Rules

### Privileged Mode vs Device Mounting

| Approach | When to Use | Command |
|----------|-------------|---------|
| **Privileged (default)** | Development, multiple NPU cards, quick setup | `--privileged` |
| Device Mounting | Production, specific cards, security required | `--device=/dev/davinci0` |

**Rule:** Default to `--privileged` unless user explicitly requests device mounting.

### Network Mode

| Mode | When to Use | Command |
|------|-------------|---------|
| **Host (default)** | Development, local access, no port conflicts | `--network host` |
| Bridge | Production, need port isolation, service deployment | `--network bridge -p 8080:80` |

**Rule:** Default to `--network host` for NPU servers.

### NPU Card Selection

- **all (default)**: Use `--privileged` for all NPU access
- **specific**: Use `--device=/dev/davinci{i}` for each card

## Common Scenarios

### Scenario 1: Development Environment (Most Common)

**User request:** "帮我启动一个开发容器" or "I want a dev container"

**Steps:**
1. Check environment (driver, NPU, Docker)
2. List available images
3. Recommend appropriate image based on task
4. Use defaults (privileged, host network, all NPUs)
5. Mount home directory

**Command:**
```bash
docker run -it -d \
  --name pytorch-dev \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /home/user:/home/user \
  -w /home/user \
  pytorch/pytorch:latest \
  /bin/bash
```

### Scenario 2: Single NPU Card Training

**User request:** "我只用卡0训练"

**Command:**
```bash
docker run -it -d \
  --name train-card0 \
  --network host \
  --device=/dev/davinci0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /data:/data \
  -w /data \
  -e ASCEND_DEVICE_ID=0 \
  ascend-pytorch:latest \
  /bin/bash
```

### Scenario 3: Multi-Card Distributed Training

**User request:** "我要用所有卡进行分布式训练"

**Command:**
```bash
docker run -it -d \
  --name distributed-train \
  --privileged \
  --network host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /home/user:/workspace \
  -w /workspace \
  --shm-size=16g \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3 \
  -e RANK_SIZE=4 \
  ascend-mindspore:latest \
  /bin/bash
```

### Scenario 4: Inference Service with Port Mapping

**User request:** "我需要启动一个推理服务，端口8080"

**Command:**
```bash
docker run -it -d \
  --name inference-svc \
  --network bridge \
  -p 8080:8000 \
  --device=/dev/davinci0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /models:/models \
  -w /models \
  ascend-inference:latest \
  /bin/bash
```

## Pre-Launch Checklist

Before executing `docker run`, verify:

- [ ] Ascend driver exists at `/usr/local/Ascend/driver/`
- [ ] NPU devices available (`/dev/davinci*`)
- [ ] Docker service is running
- [ ] Image exists locally or can be pulled
- [ ] Container name doesn't conflict with existing container
- [ ] User confirmed the command

## Error Handling

| Error | Solution |
|-------|----------|
| "driver not found" | Ask user to install Ascend driver |
| "No NPU devices" | Check hardware connection and driver |
| "Docker not running" | Start Docker: `systemctl start docker` |
| "Image not found" | Pull image or use different image |
| "Container name exists" | Remove old: `docker rm -f <name>` or use new name |
| "Permission denied" | Check user is in docker group or use sudo |

## Reference Files

- [npu-devices.md](references/npu-devices.md) - NPU device querying and mounting details
- [examples.md](references/examples.md) - More detailed examples

## Quick Reference Commands

```bash
# List local images
docker images

# Check NPU devices
ls -l /dev/davinci* && npu-smi info

# List running containers
docker ps

# Enter container
docker exec -it <name> bash

# View logs
docker logs -f <name>

# Stop container
docker stop <name>

# Remove container
docker rm <name>

# Force remove container
docker rm -f <name>
```
