# Container Lifecycle Workflows

## Overview

This document describes Docker container workflows for remote server environments, including connection, execution, and cleanup patterns.

## Remote Container Operations

### Execute Command in Remote Container

```bash
# Single command
ssh <user>@<host> "docker exec my-container ls /workspace"

# With environment variables
ssh <user>@<host> "docker exec -e MY_VAR=value my-container env"

# Interactive command
ssh -t <user>@<host> "docker exec -it my-container bash"
```

### List Containers on Remote Host

```bash
# Running containers
ssh <user>@<host> "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"

# All containers (including stopped)
ssh <user>@<host> "docker ps -a"

# Filter by name
ssh <user>@<host> "docker ps -a --filter name=my-container"
```

### Copy Files to/from Remote Container

```bash
# To container (via host as intermediary)
scp file.py <user>@<host>:/tmp/
ssh <user>@<host> "docker cp /tmp/file.py my-container:/workspace/"

# From container (via host as intermediary)
ssh <user>@<host> "docker cp my-container:/workspace/output.txt /tmp/"
scp <user>@<host>:/tmp/output.txt ./

# Direct docker cp on remote host
ssh <user>@<host> "docker cp /local/file.py my-container:/workspace/"
ssh <user>@<host> "docker cp my-container:/workspace/output.txt /local/"
```

## Lifecycle Management

### Check Container Status

```bash
# Check if running
ssh <user>@<host> "docker ps --filter name=my-container --filter status=running"

# Inspect container
ssh <user>@<host> "docker inspect my-container"

# Check logs
ssh <user>@<host> "docker logs my-container"
ssh <user>@<host> "docker logs -f --tail 100 my-container"

# Resource usage
ssh <user>@<host> "docker stats my-container --no-stream"
```

### Stop and Start Containers

```bash
# Stop container
ssh <user>@<host> "docker stop my-container"

# Start container
ssh <user>@<host> "docker start my-container"

# Restart container
ssh <user>@<host> "docker restart my-container"
```

### Remove Containers

```bash
# Remove stopped container
ssh <user>@<host> "docker rm my-container"

# Force remove running container
ssh <user>@<host> "docker rm -f my-container"

# Remove all stopped containers
ssh <user>@<host> "docker container prune"

# Remove container and volumes
ssh <user>@<host> "docker rm -v my-container"
```

## Workflow Patterns

### Pattern 1: Temporary Task Container

**Use case:** One-time task with auto-cleanup

```bash
# 1. Create container
ssh user@host "docker run -d \
  --name temp-task-$(date +%s) \
  --network host \
  -v /home/data:/workspace \
  my-image:latest \
  /bin/bash -c 'python /workspace/task.py'"

# 2. Wait for completion and get logs
ssh user@host "docker wait temp-task-xxx"
ssh user@host "docker logs temp-task-xxx"

# 3. Retrieve results
ssh user@host "docker cp temp-task-xxx:/workspace/output ./"

# 4. Cleanup
ssh user@host "docker rm temp-task-xxx"
```

### Pattern 2: Long-Running Service Container

**Use case:** Web service or API server

```bash
# 1. Create container with restart policy
ssh user@host "docker run -d \
  --name my-service \
  --network host \
  --restart unless-stopped \
  -v /home/models:/models \
  my-image:latest \
  python server.py --port 8000"

# 2. Monitor
ssh user@host "docker logs -f my-service"

# 3. Health check
ssh user@host "curl -s localhost:8000/health"
```

### Pattern 3: Development Container

**Use case:** Interactive development environment

```bash
# 1. Create or reuse container
ssh user@host "docker ps -a | grep dev-container" || \
ssh user@host "docker run -it -d \
  --name dev-container \
  --network host \
  -v /home/user:/home/user \
  -w /home/user \
  my-dev-image:latest \
  /bin/bash"

# 2. Connect for development
ssh -t user@host "docker exec -it dev-container bash"

# 3. Container persists for future use
```

### Pattern 4: Batch Processing with Cleanup

**Use case:** Process multiple files, auto-cleanup

```bash
#!/bin/bash
# batch_process.sh

HOST="user@192.168.1.100"
CONTAINER="batch-$(date +%Y%m%d-%H%M%S)"
IMAGE="my-image:latest"

# Create container
ssh $HOST "docker run -d \
  --name $CONTAINER \
  -v /home/data:/data \
  $IMAGE /bin/bash -c 'sleep infinity'"

# Process files
for file in input1.txt input2.txt input3.txt; do
  ssh $HOST "docker exec $CONTAINER python /scripts/process.py /data/$file"
done

# Collect results
ssh $HOST "docker cp $CONTAINER:/data/results ./results"

# Cleanup
ssh $HOST "docker rm -f $CONTAINER"

echo "Batch processing complete. Container $CONTAINER removed."
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
ssh user@host "docker logs my-container"

# Check for port conflicts
ssh user@host "ss -tlnp | grep <port>"

# Check image exists
ssh user@host "docker images | grep <image>"
```

### Permission Issues

```bash
# Check if privileged
ssh user@host "docker inspect --format='{{.HostConfig.Privileged}}' my-container"

# Check user in container
ssh user@host "docker exec my-container whoami"

# Run as root
ssh user@host "docker exec -u 0 my-container <command>"
```

### Disk Space Issues

```bash
# Check disk usage
ssh user@host "docker system df"

# Clean up unused resources
ssh user@host "docker system prune -f"

# Remove unused images
ssh user@host "docker image prune -f"
```
