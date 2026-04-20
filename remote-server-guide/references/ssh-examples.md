# SSH Command Examples

## Basic Connections

### Simple Connection

```bash
# Connect with password
ssh user@hostname

# Connect with specific port
ssh -p 2222 user@hostname

# Connect with SSH key
ssh -i ~/.ssh/my_key user@hostname
```

### Connection with Options

```bash
# Disable host key checking (first connection)
ssh -o StrictHostKeyChecking=no user@hostname

# Set connection timeout
ssh -o ConnectTimeout=30 user@hostname

# Verbose output for debugging
ssh -v user@hostname

# Extra verbose (levels 1-3)
ssh -vvv user@hostname
```

## Password Authentication Methods

**Important:** Standard SSH requires interactive password input. For automated password authentication, use one of these tools:

### sshpass (Command Line)

```bash
# Install sshpass
# Ubuntu/Debian:
sudo apt-get install sshpass

# CentOS/RHEL:
sudo yum install sshpass

# macOS:
brew install sshpass
```

#### Basic Usage

```bash
# Basic password auth
sshpass -p 'your_password' ssh user@hostname

# With port
sshpass -p 'your_password' ssh -p 2222 user@hostname

# Disable host key checking
sshpass -p 'your_password' ssh -o StrictHostKeyChecking=no user@hostname

# Execute command
sshpass -p 'your_password' ssh user@hostname "ls -la"

# With jump host
sshpass -p 'your_password' ssh -J jump@jump-host user@target-host
```

#### Secure Password Handling

```bash
# Use environment variable (avoids password in process list)
export SSHPASS='your_password'
sshpass -e ssh user@hostname

# Read from file
echo 'your_password' > ~/.ssh/password.txt
chmod 600 ~/.ssh/password.txt
sshpass -f ~/.ssh/password.txt ssh user@hostname

# Read from environment with fallback
PASSWORD=${REMOTE_PASSWORD:-$(cat ~/.ssh/password.txt)}
SSHPASS="$PASSWORD" sshpass -e ssh user@hostname
```

### Python Paramiko

```python
import paramiko

# Basic connection
def ssh_connect(host, port, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, password=password)
    return client

# Execute command
def ssh_exec(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    return {
        'stdout': stdout.read().decode(),
        'stderr': stderr.read().decode(),
        'exit_code': stdout.channel.recv_exit_status()
    }

# Usage
client = ssh_connect('192.168.1.100', 22, 'root', 'password')
result = ssh_exec(client, 'uptime')
print(result['stdout'])
client.close()
```

#### Paramiko with Jump Host

```python
import paramiko
from paramiko import SSHClient, ProxyCommand

# Via jump host using ProxyCommand
jump_host = 'jump.example.com'
target_host = '192.168.1.100'

proxy_cmd = f'ssh -W {target_host}:22 jump-user@{jump_host}'
sock = paramiko.ProxyCommand(proxy_cmd)

client = SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(target_host, username='user', password='password', sock=sock)
```

#### Paramiko File Transfer (SFTP)

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('hostname', username='user', password='password')

# SFTP operations
sftp = client.open_sftp()

# Upload file
sftp.put('/local/file.txt', '/remote/file.txt')

# Download file
sftp.get('/remote/file.txt', '/local/file.txt')

# List directory
for entry in sftp.listdir('/remote/path'):
    print(entry)

sftp.close()
client.close()
```

### Python Fabric (High-Level Wrapper)

```bash
# Install
pip install fabric
```

```python
from fabric import Connection
from invoke import task

# Basic connection with password
conn = Connection(
    host='192.168.1.100',
    user='root',
    connect_kwargs={'password': 'your_password'}
)

# Execute command
result = conn.run('uptime', hide=True)
print(result.stdout)

# Multiple commands
conn.run('cd /workspace && ls -la')

# File transfer
conn.put('local_file.txt', '/remote/path/')
conn.get('/remote/file.txt', 'local_path/')

# With sudo
conn.sudo('docker ps')

# Context manager
with conn.cd('/workspace'):
    conn.run('python script.py')

conn.close()
```

#### Fabric with Jump Host

```python
from fabric import Connection

# Define jump host
jump = Connection('jump-user@jump-host')

# Connect via jump host
target = Connection(
    host='target-host',
    user='target-user',
    gateway=jump,
    connect_kwargs={'password': 'password'}
)

target.run('hostname')
```

### Security Best Practices

| Method | Security | Use Case |
|--------|----------|----------|
| SSH keys | Highest | Production, always preferred |
| sshpass + env var | Medium-high | Automation scripts |
| sshpass + file | High | Secure storage |
| sshpass -p | Low | Testing only (password visible in ps) |
| Paramiko | High | Python applications |
| Fabric | High | Python deployment scripts |

```bash
# WARNING: Password visible in process list
sshpass -p 'secret' ssh user@host

# BETTER: Use environment variable
export SSHPASS='secret'
sshpass -e ssh user@host

# BEST: Use SSH keys
ssh -i ~/.ssh/id_rsa user@host
```

## Jump Host (Bastion)

### Using -J Flag

```bash
# Single jump host
ssh -J jump-user@jump-host user@target-host

# Multiple jump hosts
ssh -J jump1,jump2 user@target-host

# Jump host with port
ssh -J jump-user@jump-host:2222 user@target-host:22
```

### Using ProxyCommand

```bash
# Legacy method
ssh -o ProxyCommand="ssh -W %h:%p jump-user@jump-host" user@target-host
```

## Remote Command Execution

### Single Command

```bash
# Execute single command
ssh user@host "ls -la"

# Command with pipes
ssh user@host "cat /var/log/syslog | grep error"

# Command with quotes
ssh user@host "echo 'Hello World'"
```

### Multiple Commands

```bash
# Using semicolons
ssh user@host "cd /tmp && ls -la"

# Using here-doc
ssh user@host << 'EOF'
  cd /tmp
  ls -la
  echo "Done"
EOF
```

### Interactive Commands

```bash
# Force pseudo-terminal allocation
ssh -t user@host "top"

# Interactive shell
ssh -t user@host "docker exec -it my-container bash"
```

## File Transfer

### SCP Commands

```bash
# Copy file to remote
scp local-file.txt user@host:/remote/path/

# Copy file from remote
scp user@host:/remote/file.txt ./local/path/

# Copy directory recursively
scp -r local-dir/ user@host:/remote/path/

# With specific port
scp -P 2222 local-file.txt user@host:/remote/path/
```

### Rsync Commands

```bash
# Sync directory to remote
rsync -avz local-dir/ user@host:/remote/dir/

# Sync from remote
rsync -avz user@host:/remote/dir/ ./local-dir/

# Dry run (preview changes)
rsync -avzn local-dir/ user@host:/remote/dir/

# Exclude patterns
rsync -avz --exclude '*.log' local-dir/ user@host:/remote/dir/

# With progress
rsync -avz --progress local-dir/ user@host:/remote/dir/

# Via jump host
rsync -avz -e "ssh -J jump@jump-host" local-dir/ user@target:/remote/
```

## Background Execution

### Nohup

```bash
# Run in background
ssh user@host "nohup python script.py > output.log 2>&1 &"

# Check background process
ssh user@host "ps aux | grep script.py"
```

### Screen/Tmux

```bash
# Start screen session
ssh user@host "screen -dmS mysession bash -c 'python script.py'"

# List screen sessions
ssh user@host "screen -ls"

# Attach to screen
ssh -t user@host "screen -r mysession"

# Using tmux
ssh user@host "tmux new-session -d 'python script.py'"
ssh -t user@host "tmux attach"
```

## Port Forwarding

### Local Port Forwarding

```bash
# Forward local port to remote
ssh -L 8080:localhost:80 user@host

# Access remote service via localhost:8080
```

### Remote Port Forwarding

```bash
# Forward remote port to local
ssh -R 8080:localhost:80 user@host
```

### Dynamic Port Forwarding (SOCKS)

```bash
# Create SOCKS proxy
ssh -D 1080 user@host
```

## SSH Config Examples

### Basic Config

```
# ~/.ssh/config

Host myserver
    HostName 192.168.1.100
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

### With Jump Host

```
Host jump
    HostName jump.example.com
    User jump-user
    IdentityFile ~/.ssh/jump_key

Host target
    HostName 10.0.0.50
    User admin
    ProxyJump jump
    IdentityFile ~/.ssh/target_key
```

### With Custom Options

```
Host server-*
    User root
    Port 22
    StrictHostKeyChecking no
    ConnectTimeout 30
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Troubleshooting

```bash
# Test connectivity
ping hostname && nc -zv hostname 22

# Check SSH service
ssh user@host "systemctl status sshd"

# Fix key permissions
chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_rsa

# Remove old host key
ssh-keygen -R hostname
```
