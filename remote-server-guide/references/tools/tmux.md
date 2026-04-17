# tmux Interactive SSH Method

## Overview

tmux provides a secure way to establish SSH connections where the user inputs their password directly, without the AI ever seeing or handling credentials. This is the most secure method for password-based SSH authentication when SSH keys are not available.

## Why Use tmux for SSH?

| Benefit | Description |
|---------|-------------|
| **Maximum Security** | Password is never exposed to AI or logged |
| **No Credential Storage** | No need to save passwords anywhere |
| **User Control** | User has full control over authentication |
| **Works Everywhere** | Only requires tmux to be installed |

## Basic Workflow

```
1. AI creates tmux session with SSH command
2. AI tells user to attach to the session
3. User enters password in their terminal
4. AI sends commands and retrieves output
5. AI kills session when done
```

## Commands Reference

### Session Management

```bash
# Create a new tmux session with SSH command
tmux new-session -d -s <session_name> "ssh -o StrictHostKeyChecking=no <user>@<host> -p <port>"

# List all tmux sessions
tmux list-sessions

# Check if a session exists
tmux has-session -t <session_name> 2>/dev/null && echo "exists" || echo "not found"

# Kill a session
tmux kill-session -t <session_name>

# Kill all sessions
tmux kill-server
```

### User Interaction

```bash
# Tell user to attach (they run this in their terminal)
tmux attach -t <session_name>

# Or with short form
tmux a -t <session_name>

# Detach from session (user presses this key combination)
# Ctrl+B, then D
```

### Command Execution

```bash
# Send a command to the session
tmux send-keys -t <session_name> "<command>" Enter

# Send multiple commands
tmux send-keys -t <session_name> "cd /workspace" Enter
tmux send-keys -t <session_name> "ls -la" Enter

# Send special keys
tmux send-keys -t <session_name> C-c  # Ctrl+C
tmux send-keys -t <session_name> C-d  # Ctrl+D (logout)
```

### Output Capture

```bash
# Capture current screen content
tmux capture-pane -t <session_name> -p

# Capture with line numbers
tmux capture-pane -t <session_name> -p | cat -n

# Capture more lines (scrollback)
tmux capture-pane -t <session_name> -p -S -100  # Last 100 lines

# Capture to file
tmux capture-pane -t <session_name> -p > /tmp/output.txt
```

## Complete Example

### Step-by-Step SSH Session

```bash
# 1. Create session
tmux new-session -d -s remote_session "ssh -o StrictHostKeyChecking=no root@192.168.1.100"

# 2. Inform user
echo "Please run in another terminal: tmux attach -t remote_session"
echo "Enter your password when prompted."

# 3. Wait for user to connect (they will confirm)
# ... user attaches and enters password ...

# 4. Verify connection by checking output
tmux capture-pane -t remote_session -p

# 5. Execute commands
tmux send-keys -t remote_session "hostname" Enter

# 6. Get output after command completes
tmux capture-pane -t remote_session -p

# 7. Execute more commands
tmux send-keys -t remote_session "docker ps" Enter
tmux capture-pane -t remote_session -p

# 8. Exit SSH
tmux send-keys -t remote_session "exit" Enter

# 9. Clean up
tmux kill-session -t remote_session
```

## Working with Docker Containers

```bash
# After SSH connection is established via tmux

# List containers
tmux send-keys -t remote_session "docker ps -a" Enter
tmux capture-pane -t remote_session -p

# Execute in container
tmux send-keys -t remote_session "docker exec -it my-container bash" Enter

# Now in container - run commands
tmux send-keys -t remote_session "python inference.py" Enter
tmux capture-pane -t remote_session -p

# Exit container
tmux send-keys -t remote_session "exit" Enter
```

## Handling Long-Running Commands

For commands that take a long time (like training or batch processing):

```bash
# Start the long-running command
tmux send-keys -t remote_session "python train.py --epochs 100" Enter

# Check status periodically
tmux capture-pane -t remote_session -p | tail -20

# If you need to wait for completion, check for prompt return
# The prompt appearing again indicates command completion
```

## Troubleshooting

### Session Already Exists

```bash
# Error: session name already in use
# Solution: kill the old session first
tmux kill-session -t <session_name> 2>/dev/null
tmux new-session -d -s <session_name> "ssh ..."
```

### Connection Timeout

```bash
# Add connection timeout
tmux new-session -d -s remote_session "ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@host"
```

### SSH Key Fingerprint Prompt

```bash
# The -o StrictHostKeyChecking=no option handles this
# But for first connection, you might need:
tmux new-session -d -s remote_session "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@host"
```

### Session Stuck

```bash
# Send Ctrl+C to interrupt
tmux send-keys -t remote_session C-c

# Or kill and recreate
tmux kill-session -t remote_session
tmux new-session -d -s remote_session "ssh ..."
```

## Security Considerations

1. **Session Names**: Use descriptive but non-sensitive names
2. **Cleanup**: Always kill sessions when done
3. **Screen Content**: Password prompts are visible in tmux, but not logged
4. **Concurrent Access**: Only one user should attach at a time for password input

## Comparison with Other Methods

| Feature | tmux | sshpass | paramiko |
|---------|------|---------|----------|
| Password visible to AI | No | Yes | Yes |
| Password logged | No | Yes (process list) | No |
| File transfer | Manual (scp) | Yes (scp) | Yes (SFTP) |
| Setup required | tmux install | sshpass install | pip install |
| User interaction | Required | None | None |
| Best for | One-time secure access | Scripts | Python apps |

## Tips for AI Assistants

1. **Always check if tmux is available first**: `which tmux`
2. **Use meaningful session names**: `remote_session`, `dev_server`, etc.
3. **Wait for user confirmation** before sending commands after they attach
4. **Capture output after each command** to verify results
5. **Clean up sessions** when the task is complete
6. **Handle session conflicts** by killing old sessions before creating new ones
