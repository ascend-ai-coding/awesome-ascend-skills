---
name: remote-server-guide
description: "Guide for connecting to remote servers via SSH, managing containers, and executing commands remotely. Covers SSH connection setup with multiple authentication methods (SSH key, sshpass, Paramiko, Fabric, tmux interactive), Docker container connection and command execution, file transfer, and troubleshooting. Use this skill whenever users need to: (1) Connect to a remote server or VPS via SSH, (2) Execute commands on remote machines, (3) Transfer files to/from remote servers, (4) Connect to Docker containers on remote hosts, (5) Set up SSH password authentication tools. Even if the user just says 'help me run something on my server' or 'connect to my machine', this skill applies."
---

# Remote Server Guide

A workflow for connecting to remote servers, executing commands, and managing containers over SSH. This skill is tool-agnostic — it helps you pick the right SSH tool for the job and walks through the full lifecycle from connection to cleanup.

## When to Use This Skill

- User needs to connect to a remote server (cloud VM, on-prem server, dev machine, etc.)
- User wants to run commands on a remote host
- User needs to connect to a Docker container on a remote server
- User wants to transfer files to/from a remote server
- User is having trouble with SSH authentication

## Authentication Methods

| Method | Security | Convenience | Best For |
|--------|----------|-------------|----------|
| **SSH Key** | Highest | High | Production, frequent use |
| **tmux (interactive)** | High (password never exposed to AI) | Medium | One-time access, security-conscious users |
| **sshpass** | Medium (password in command) | High | Quick scripts, trusted environment |
| **paramiko/fabric** | Medium | Medium | Python automation |

## Workflow

### Phase 1: Gather Connection Details

Before connecting, collect from the user:

1. **Host address** (IP or hostname)
2. **SSH port** (default: 22)
3. **Username** (e.g., root, ubuntu, user)
4. **Authentication method**:
   - SSH key (recommended)
   - Password (requires tool selection — see Phase 2)
5. **Jump host** (optional, for bastion/proxy setups)

### Phase 2: Set Up Authentication

#### SSH Key (preferred)

```bash
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i <key-path> <user>@<host> -p <port> "echo 'Connection successful'"
```

#### Password Authentication

If the user needs password auth, first detect available tools:

```bash
which sshpass >/dev/null 2>&1 && echo "sshpass: available" || echo "sshpass: not found"
python3 -c "import paramiko" 2>/dev/null && echo "paramiko: available" || echo "paramiko: not found"
python3 -c "import fabric" 2>/dev/null && echo "fabric: available" || echo "fabric: not found"
which tmux >/dev/null 2>&1 && echo "tmux: available" || echo "tmux: not found"
```

**If the user is concerned about password security**, always recommend **tmux interactive** — the password is typed by the user directly and never seen by the AI.

**Tool selection guidance:**

| Scenario | Recommended Tool | Reason |
|----------|------------------|--------|
| Security-conscious user | tmux | Password never exposed to AI |
| One-time access | tmux | No credential storage needed |
| Quick shell script | sshpass | Simple, no dependencies |
| Python application | paramiko | Full control, lower level |
| Deployment automation | fabric | High-level API, cleaner code |
| Need SFTP | paramiko or fabric | Built-in support |
| Interactive terminal | tmux or sshpass | Native TTY support |

**If no tools are available**, offer to install one:

```bash
# tmux (Ubuntu/Debian)
sudo apt-get install -y tmux
# sshpass (Ubuntu/Debian)
sudo apt-get install -y sshpass
# paramiko
pip install paramiko
# fabric
pip install fabric
```

### Phase 3: Test Connection

Verify the connection works before proceeding.

**sshpass:**
```bash
sshpass -p '<password>' ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no <user>@<host> -p <port> "echo 'Connection successful'"
```

**tmux (interactive):**
```bash
tmux new-session -d -s remote_session "ssh -o StrictHostKeyChecking=no <user>@<host> -p <port>"
# Tell user: "Please run: tmux attach -t remote_session and enter your password"
# After user confirms, verify:
tmux capture-pane -t remote_session -p
```

**paramiko:**
```python
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='<host>', port=<port>, username='<user>', password='<password>')
print('Connection successful')
client.close()
```

**fabric:**
```python
from fabric import Connection
conn = Connection(host='<host>', user='<user>', connect_kwargs={'password': '<password>'})
conn.run('echo "Connection successful"', hide=True)
conn.close()
```

If connection fails, troubleshoot:
- Check host address and port
- Verify credentials
- Check network/firewall/VPN
- Verify jump host configuration (if used)

### Phase 4: Choose Execution Environment

Ask the user where commands will run:

1. **Directly on host** — most common, no extra setup
2. **Inside an existing Docker container** — connect to a running container
3. **Inside a new Docker container** — create a container first (for specialized workloads, consider hardware-specific skills like `npu-docker-launcher` for NPU servers)

#### Connecting to an Existing Container

**Important: tmux vs other tools behave very differently for container work.**

**tmux (persistent session):** Enter the container shell once, then send all subsequent commands directly — they execute inside the container automatically. The session maintains state (working directory, environment variables, shell context).

```bash
# List containers
tmux send-keys -t remote_session "docker ps" Enter
tmux capture-pane -t remote_session -p

# Enter container ONCE — all subsequent commands run inside the container
tmux send-keys -t remote_session "docker exec -it <container-name> bash" Enter

# Now you are INSIDE the container — just send commands directly, no need to repeat docker exec
tmux send-keys -t remote_session "cd /workspace" Enter
tmux send-keys -t remote_session "python inference.py" Enter
tmux send-keys -t remote_session "cat result.txt" Enter
tmux capture-pane -t remote_session -p

# Exit container when done (returns to host shell)
tmux send-keys -t remote_session "exit" Enter
```

**Other tools (stateless):** Each command is a separate SSH connection with no shared state. You must prefix every command with `docker exec`:

```bash
# Every command needs the full docker exec prefix
ssh <user>@<host> "docker exec <container-name> ls /workspace"
ssh <user>@<host> "docker exec <container-name> python inference.py"
ssh <user>@<host> "docker exec <container-name> cat result.txt"
```

### Phase 5: Execute Commands

Use the selected SSH tool consistently for all operations.

#### Single Command

**tmux:** Since tmux maintains a persistent shell session, just send the command directly. If you previously entered a container, the command runs inside the container.

```bash
tmux send-keys -t remote_session "<command>" Enter
tmux capture-pane -t remote_session -p
```

**Other tools (stateless, each command is independent):**

| Tool | On host | In container |
|------|---------|--------------|
| SSH key | `ssh -i <key> <user>@<host> "<command>"` | `ssh -i <key> <user>@<host> "docker exec <container> <command>"` |
| sshpass | `sshpass -p '<pwd>' ssh <user>@<host> "<command>"` | `sshpass -p '<pwd>' ssh <user>@<host> "docker exec <container> <command>"` |
| paramiko | `client.exec_command("<command>")` | `client.exec_command("docker exec <container> <command>")` |
| fabric | `conn.run("<command>")` | `conn.run("docker exec <container> <command>")` |

#### File Transfer

| Tool | Upload | Download |
|------|--------|----------|
| scp | `scp <local> <user>@<host>:<remote>` | `scp <user>@<host>:<remote> <local>` |
| sshpass+scp | `sshpass -p '<pwd>' scp <local> <user>@<host>:<remote>` | `sshpass -p '<pwd>' scp <user>@<host>:<remote> <local>` |
| paramiko | `sftp.put('local', 'remote')` | `sftp.get('remote', 'local')` |
| fabric | `conn.put('local', 'remote/')` | `conn.get('remote', 'local')` |

#### Multi-Step Script

For complex operations, upload a script and execute it:

```bash
# Upload script
scp setup.sh <user>@<host>:/tmp/
# Execute
ssh <user>@<host> "bash /tmp/setup.sh"
# Cleanup
ssh <user>@<host> "rm /tmp/setup.sh"
```

### Phase 6: Cleanup

When done, clean up resources:

- Close SSH connections (`client.close()`, `conn.close()`)
- Kill tmux sessions (`tmux kill-session -t remote_session`)
- Stop/remove temporary containers if applicable

---

## Operations Reference

### tmux (Interactive)

```bash
# Create SSH session (user inputs password)
tmux new-session -d -s remote_session "ssh -o StrictHostKeyChecking=no <user>@<host> -p <port>"
# User runs: tmux attach -t remote_session

# Execute command after connection
tmux send-keys -t remote_session "<command>" Enter

# Get output
tmux capture-pane -t remote_session -p

# Kill session when done
tmux kill-session -t remote_session
```

### sshpass

```bash
# Execute command
sshpass -p '<password>' ssh <user>@<host> "<command>"

# Upload file
sshpass -p '<password>' scp <local> <user>@<host>:<remote>

# Docker exec on remote
sshpass -p '<password>' ssh <user>@<host> "docker exec <container> <command>"
```

### paramiko

```python
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='<host>', username='<user>', password='<password>')

# Execute command
stdin, stdout, stderr = client.exec_command('<command>')
print(stdout.read().decode())

# SFTP
sftp = client.open_sftp()
sftp.put('local.txt', '/remote/path/remote.txt')
sftp.get('/remote/file.txt', 'local.txt')

client.close()
```

### fabric

```python
from fabric import Connection
conn = Connection(host='<host>', user='<user>',
                  connect_kwargs={'password': '<password>'})

result = conn.run('<command>', hide=True)
print(result.stdout)

conn.put('local.txt', '/remote/path/')
conn.get('/remote/file.txt', 'local.txt')

conn.close()
```

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| Connection refused | SSH not running | Check SSH service on remote |
| Permission denied | Wrong credentials | Verify username/password/key |
| Host key verification failed | First connection | Add `-o StrictHostKeyChecking=no` |
| Network unreachable | Network issue | Check firewall, VPN |
| Tool not found | Not installed | Run installation command |
| Container not found | Wrong name | `docker ps -a` to list all |

## Security Best Practices

1. **Prefer SSH keys** over passwords when possible
2. **Avoid hardcoding passwords** — use environment variables or `SSHPASS` env var
3. **Restrict key permissions**: `chmod 600 ~/.ssh/id_rsa`
4. **Use tmux method** if you don't want the AI to handle your password
5. **Clean up** temporary containers and sessions when done

## Detailed Reference Files

For in-depth usage of each tool, consult:

- [tools/tmux.md](references/tools/tmux.md) — tmux interactive SSH (most secure)
- [tools/sshpass.md](references/tools/sshpass.md) — sshpass detailed usage
- [tools/paramiko-basic.md](references/tools/paramiko-basic.md) — Paramiko basic operations
- [tools/paramiko-advanced.md](references/tools/paramiko-advanced.md) — Paramiko Docker and jump host
- [tools/fabric-basic.md](references/tools/fabric-basic.md) — Fabric basic operations
- [tools/fabric-advanced.md](references/tools/fabric-advanced.md) — Fabric Docker and parallel execution
- [tool-comparison.md](references/tool-comparison.md) — Tool comparison and selection guide
- [ssh-examples.md](references/ssh-examples.md) — General SSH command examples
- [container-workflows.md](references/container-workflows.md) — Container lifecycle workflows
