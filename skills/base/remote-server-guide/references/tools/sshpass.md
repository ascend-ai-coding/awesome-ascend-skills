# sshpass Tool Reference

## Overview

sshpass is a command-line utility for non-interactively providing passwords to SSH. It's the simplest tool for password-based SSH authentication in shell scripts.

## Installation

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y sshpass

# CentOS/RHEL/AlmaLinux
sudo yum install -y epel-release && sudo yum install -y sshpass

# Fedora
sudo dnf install -y sshpass

# macOS
brew install hudochenkov/sshpass/sshpass

# Alpine Linux
apk add sshpass

# Arch Linux
sudo pacman -S sshpass

# Verify installation
sshpass -V
```

## Basic Usage

### Connection

```bash
# Basic connection with password
sshpass -p 'your_password' ssh user@hostname

# With specific port
sshpass -p 'your_password' ssh -p 2222 user@hostname

# Disable host key checking (first connection)
sshpass -p 'your_password' ssh -o StrictHostKeyChecking=no user@hostname

# With connection timeout
sshpass -p 'your_password' ssh -o ConnectTimeout=30 user@hostname
```

### Secure Password Handling

```bash
# Method 1: Environment variable (recommended)
export SSHPASS='your_password'
sshpass -e ssh user@hostname "command"
unset SSHPASS  # Clean up after use

# Method 2: Password file
echo 'your_password' > ~/.ssh/remote_password
chmod 600 ~/.ssh/remote_password
sshpass -f ~/.ssh/remote_password ssh user@hostname "command"

# Method 3: Read from stdin
echo 'your_password' | sshpass ssh user@hostname "command"

# Method 4: From variable
PASSWORD=$(cat ~/.ssh/remote_password)
SSHPASS="$PASSWORD" sshpass -e ssh user@hostname "command"
```

## Remote Command Execution

### Single Command

```bash
# Execute command
sshpass -p 'password' ssh user@host "ls -la"

# Command with pipes
sshpass -p 'password' ssh user@host "cat /var/log/syslog | grep error"

# Capture output
RESULT=$(sshpass -p 'password' ssh user@host "hostname")
echo "Remote host: $RESULT"

# Check exit code
sshpass -p 'password' ssh user@host "exit 1"
echo "Exit code: $?"
```

### Multiple Commands

```bash
# Using semicolons
sshpass -p 'password' ssh user@host "cd /tmp && ls -la && pwd"

# Using here-doc for complex scripts
sshpass -p 'password' ssh user@host << 'EOF'
    cd /workspace
    source venv/bin/activate
    python script.py
    echo "Done"
EOF

# Execute local script on remote
sshpass -p 'password' ssh user@host "bash -s" < local_script.sh
```

### Interactive Commands

```bash
# Force pseudo-terminal allocation
sshpass -p 'password' ssh -t user@host "top"

# Interactive shell
sshpass -p 'password' ssh -t user@host "bash"

# Interactive Python
sshpass -p 'password' ssh -t user@host "python3"
```

## File Transfer

### SCP (Secure Copy)

```bash
# Upload file
sshpass -p 'password' scp local_file.txt user@host:/remote/path/

# Download file
sshpass -p 'password' scp user@host:/remote/file.txt ./local/path/

# Upload directory recursively
sshpass -p 'password' scp -r local_dir/ user@host:/remote/path/

# Download directory
sshpass -p 'password' scp -r user@host:/remote/dir/ ./local/

# With specific port
sshpass -p 'password' scp -P 2222 local_file.txt user@host:/remote/

# Preserve file attributes
sshpass -p 'password' scp -p local_file.txt user@host:/remote/

# Show progress
sshpass -p 'password' scp -v local_file.txt user@host:/remote/
```

### Rsync (Recommended for Large Files)

```bash
# Sync directory to remote
sshpass -p 'password' rsync -avz local_dir/ user@host:/remote/dir/

# Sync from remote
sshpass -p 'password' rsync -avz user@host:/remote/dir/ ./local-dir/

# Dry run (preview changes)
sshpass -p 'password' rsync -avzn local_dir/ user@host:/remote/dir/

# Exclude patterns
sshpass -p 'password' rsync -avz --exclude '*.log' --exclude '*.tmp' \
    local_dir/ user@host:/remote/dir/

# Show progress
sshpass -p 'password' rsync -avz --progress local_dir/ user@host:/remote/dir/

# Delete files that don't exist locally
sshpass -p 'password' rsync -avz --delete local_dir/ user@host:/remote/dir/

# Via jump host
sshpass -p 'password' rsync -avz -e "ssh -J jump@jump-host" \
    local_dir/ user@target:/remote/
```

## Docker Operations

### Container Management

```bash
# List containers
sshpass -p 'password' ssh user@host "docker ps -a"

# List containers with format
sshpass -p 'password' ssh user@host "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"

# Create container
sshpass -p 'password' ssh user@host "docker run -d --name my-container \
    --network host \
    -v /home/data:/workspace \
    my-image:latest /bin/bash -c 'sleep infinity'"

# Start container
sshpass -p 'password' ssh user@host "docker start my-container"

# Stop container
sshpass -p 'password' ssh user@host "docker stop my-container"

# Remove container
sshpass -p 'password' ssh user@host "docker rm my-container"

# Force remove running container
sshpass -p 'password' ssh user@host "docker rm -f my-container"
```

### Container Execution

```bash
# Execute command in container
sshpass -p 'password' ssh user@host "docker exec my-container ls /workspace"

# Execute with environment variable
sshpass -p 'password' ssh user@host "docker exec -e MY_VAR=value my-container env"

# Interactive shell in container
sshpass -p 'password' ssh -t user@host "docker exec -it my-container bash"

# Execute as root
sshpass -p 'password' ssh user@host "docker exec -u 0 my-container whoami"

# Execute Python script
sshpass -p 'password' ssh user@host "docker exec my-container python /workspace/script.py"
```

### File Copy with Containers

```bash
# Copy file to container
sshpass -p 'password' ssh user@host "docker cp /local/file.py my-container:/workspace/"

# Copy file from container
sshpass -p 'password' ssh user@host "docker cp my-container:/workspace/output.txt /local/"

# Copy local file to remote container (two-step)
sshpass -p 'password' scp local_file.py user@host:/tmp/
sshpass -p 'password' ssh user@host "docker cp /tmp/local_file.py my-container:/workspace/"

# Copy from container to local (two-step)
sshpass -p 'password' ssh user@host "docker cp my-container:/workspace/output.txt /tmp/"
sshpass -p 'password' scp user@host:/tmp/output.txt ./
```

### Container Logs

```bash
# View logs
sshpass -p 'password' ssh user@host "docker logs my-container"

# Follow logs
sshpass -p 'password' ssh user@host "docker logs -f my-container"

# Last N lines
sshpass -p 'password' ssh user@host "docker logs --tail 100 my-container"

# With timestamps
sshpass -p 'password' ssh user@host "docker logs -t my-container"
```

## Jump Host

```bash
# Via jump host
sshpass -p 'password' ssh -J jump-user@jump-host user@target-host "command"

# Jump host with port
sshpass -p 'password' ssh -J jump-user@jump-host:2222 user@target-host:22

# Multiple jump hosts
sshpass -p 'password' ssh -J jump1,jump2 user@target-host

# Jump host with different credentials (nested sshpass)
# Note: This is complex - consider using paramiko or fabric instead
```

## Background Execution

### Using nohup

```bash
# Run in background
sshpass -p 'password' ssh user@host "nohup python script.py > output.log 2>&1 &"

# Check background process
sshpass -p 'password' ssh user@host "ps aux | grep script.py"

# Check output
sshpass -p 'password' ssh user@host "tail -f output.log"
```

### Using Screen

```bash
# Start screen session with command
sshpass -p 'password' ssh user@host "screen -dmS mysession bash -c 'python script.py'"

# List screen sessions
sshpass -p 'password' ssh user@host "screen -ls"

# Attach to screen (interactive)
sshpass -p 'password' ssh -t user@host "screen -r mysession"

# Kill screen session
sshpass -p 'password' ssh user@host "screen -X -S mysession quit"
```

### Using Tmux

```bash
# Start tmux session
sshpass -p 'password' ssh user@host "tmux new-session -d -s mysession 'python script.py'"

# List tmux sessions
sshpass -p 'password' ssh user@host "tmux list-sessions"

# Attach to tmux (interactive)
sshpass -p 'password' ssh -t user@host "tmux attach -t mysession"

# Kill tmux session
sshpass -p 'password' ssh user@host "tmux kill-session -t mysession"
```

## Complete Script Examples

### Example 1: Temporary Container Task

```bash
#!/bin/bash
# temp_container_task.sh

HOST="192.168.1.100"
USER="root"
PASSWORD_FILE="$HOME/.ssh/remote_password"
CONTAINER="task-$(date +%s)"

# Create container
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "
    docker run -d \
        --name $CONTAINER \
        --network host \
        -v /home/data:/workspace \
        my-image:latest \
        /bin/bash -c 'sleep infinity'
"

# Upload script
sshpass -f "$PASSWORD_FILE" scp inference.py $USER@$HOST:/tmp/
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker cp /tmp/inference.py $CONTAINER:/workspace/"

# Execute task
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker exec $CONTAINER python /workspace/inference.py"

# Get results
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker cp $CONTAINER:/workspace/output.csv /tmp/"
sshpass -f "$PASSWORD_FILE" scp $USER@$HOST:/tmp/output.csv ./

# Cleanup
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker rm -f $CONTAINER"

echo "Task completed!"
```

### Example 2: Multi-Host Deployment

```bash
#!/bin/bash
# multi_host_deploy.sh

HOSTS=("192.168.1.100" "192.168.1.101" "192.168.1.102")
USER="root"
PASSWORD_FILE="$HOME/.ssh/remote_password"

for HOST in "${HOSTS[@]}"; do
    echo "Deploying to $HOST..."

    # Check server status
    sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "uptime"

    # Pull latest image
    sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker pull my-image:latest"

    # Restart container
    sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "
        docker rm -f my-service || true
        docker run -d --name my-service \
            --network host \
            --restart unless-stopped \
            my-image:latest \
            python server.py
    "

    echo "$HOST deployed!"
done
```

### Example 3: Health Check Script

```bash
#!/bin/bash
# health_check.sh

HOST="192.168.1.100"
USER="root"
PASSWORD_FILE="$HOME/.ssh/remote_password"

echo "=== Health Check: $HOST ==="

# Check SSH connection
if sshpass -f "$PASSWORD_FILE" ssh -o ConnectTimeout=5 $USER@$HOST "echo OK" &>/dev/null; then
    echo "[OK] SSH connection"
else
    echo "[FAIL] SSH connection"
    exit 1
fi

# Check disk space
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "df -h / | tail -1" | awk '{
    usage=$5; gsub(/%/, "", usage);
    if (usage < 80) print "[OK] Disk usage: "$5;
    else print "[WARN] Disk usage: "$5;
}'

# Check Docker
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker info" &>/dev/null && \
    echo "[OK] Docker" || echo "[FAIL] Docker"

# Check service container
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "docker ps | grep my-service" &>/dev/null && \
    echo "[OK] Service container running" || echo "[WARN] Service container not running"

# Check service health endpoint
sshpass -f "$PASSWORD_FILE" ssh $USER@$HOST "curl -s localhost:8000/health" | grep -q "ok" && \
    echo "[OK] Service responding" || echo "[WARN] Service not responding"
```

## Error Handling

### Common Errors

```bash
# Check if command succeeded
if sshpass -p 'password' ssh user@host "command"; then
    echo "Success"
else
    echo "Failed"
fi

# Capture stderr
RESULT=$(sshpass -p 'password' ssh user@host "command" 2>&1)
if [ $? -ne 0 ]; then
    echo "Error: $RESULT"
fi

# Timeout handling
sshpass -p 'password' ssh -o ConnectTimeout=10 user@host "command" || {
    echo "Connection timed out"
    exit 1
}
```

## Security Best Practices

1. **Never use `-p` with password in production scripts**
   ```bash
   # Bad - password visible in process list
   sshpass -p 'secret' ssh user@host

   # Good - use environment variable
   export SSHPASS='secret'
   sshpass -e ssh user@host
   ```

2. **Use password files with restricted permissions**
   ```bash
   echo 'password' > ~/.ssh/remote_password
   chmod 600 ~/.ssh/remote_password
   sshpass -f ~/.ssh/remote_password ssh user@host
   ```

3. **Clean up sensitive data**
   ```bash
   unset SSHPASS
   rm -f /tmp/password_file
   ```

4. **Consider SSH keys when possible**
   ```bash
   # If keys are available, no need for sshpass
   ssh -i ~/.ssh/id_rsa user@host
   ```
