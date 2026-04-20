# Fabric Basic Operations

## Overview

Fabric is a high-level Python library for remote execution, built on top of Paramiko. It provides a clean, readable API and is ideal for deployment automation and configuration management.

## Installation

```bash
# pip (recommended)
pip install fabric

# pip with specific version
pip install fabric==3.2.2

# conda
conda install -c conda-forge fabric

# Verify installation
python3 -c "from fabric import Connection; print('Fabric installed')"
```

## Connection

### Simple Connection

```python
from fabric import Connection

# Basic connection with password
conn = Connection(
    host='192.168.1.100',
    user='root',
    connect_kwargs={'password': 'your_password'}
)

# Execute command
result = conn.run('hostname', hide=True)
print(result.stdout)

# Close connection
conn.close()
```

### Context Manager

```python
from fabric import Connection

# Using context manager for auto-cleanup
with Connection(host='host', user='user',
                connect_kwargs={'password': 'password'}) as conn:
    conn.run('ls -la')
    # Auto-closes on exit
```

### SSH Key Authentication

```python
from fabric import Connection

# Using SSH key
conn = Connection(
    host='192.168.1.100',
    user='root',
    connect_kwargs={'key_filename': '/path/to/private_key'}
)

# Or let Fabric find keys automatically
conn = Connection(host='host', user='user')
```

### Connection with All Options

```python
from fabric import Connection

conn = Connection(
    host='192.168.1.100',
    user='root',
    port=22,
    connect_kwargs={
        'password': 'password',
        'timeout': 30,
        'banner_timeout': 30,
        'auth_timeout': 30,
    },
    config={  # Override SSH config
        'connect_timeout': '30',
        'server_alive_interval': '60',
    }
)
```

## Remote Command Execution

### Basic Execution

```python
from fabric import Connection

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

# Execute command
result = conn.run('ls -la')

# Access output
print(result.stdout)
print(result.stderr)
print(result.exited)  # Exit code
print(result.ok)      # True if exit code == 0
print(result.failed)  # True if exit code != 0

conn.close()
```

### Hiding Output

```python
# Hide all output
result = conn.run('ls -la', hide=True)

# Hide only stdout
result = conn.run('ls -la', hide='stdout')

# Hide only stderr
result = conn.run('ls -la', hide='stderr')

# Show output only on failure
result = conn.run('ls -la', hide=True, warn=True)
if result.failed:
    print(result.stderr)
```

### Warning on Failure

```python
# Don't raise exception on failure
result = conn.run('ls /nonexistent', warn=True, hide=True)

if result.failed:
    print(f"Command failed: {result.stderr}")
else:
    print(result.stdout)
```

### Sudo Commands

```python
# Run with sudo
result = conn.sudo('docker ps')

# Sudo with password (if different from user password)
result = conn.sudo('docker ps', password='sudo_password')

# Sudo with user
result = conn.sudo('command', user='other_user')
```

### Multiple Commands

```python
from fabric import Connection

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

# Execute multiple commands
conn.run('cd /workspace')
conn.run('ls -la')
conn.run('python script.py')

# Using context manager for directory
with conn.cd('/workspace'):
    conn.run('ls -la')
    conn.run('python script.py')

# Using prefix for multiple commands
with conn.prefix('source venv/bin/activate'):
    conn.run('pip list')
    conn.run('python script.py')

conn.close()
```

### Environment Variables

```python
from fabric import Connection

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

# Set environment variable
conn.run('export MY_VAR=value && echo $MY_VAR')

# Or use context manager
with conn.env_vars({'MY_VAR': 'value', 'OTHER_VAR': 'test'}):
    conn.run('echo $MY_VAR')
    conn.run('echo $OTHER_VAR')

conn.close()
```

## File Transfer

### Upload Files

```python
from fabric import Connection

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

# Upload file to specific path
conn.put('local_file.txt', '/remote/path/file.txt')

# Upload to home directory
conn.put('local_file.txt')

# Upload to directory (keeps filename)
conn.put('local_file.txt', '/remote/path/')

conn.close()
```

### Download Files

```python
from fabric import Connection

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

# Download file to specific path
conn.get('/remote/file.txt', 'local_file.txt')

# Download to current directory (keeps filename)
conn.get('/remote/file.txt')

# Download to directory
conn.get('/remote/file.txt', './local_dir/')

conn.close()
```

### Directory Transfer

```python
from fabric import Connection
import os

def upload_dir(conn, local_dir, remote_dir):
    """Upload directory recursively"""
    for root, dirs, files in os.walk(local_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path) if rel_path != '.' else remote_dir

        # Create remote directory
        conn.run(f'mkdir -p {remote_path}', hide=True, warn=True)

        # Upload files
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file)
            conn.put(local_file, remote_file)
            print(f"Uploaded: {local_file} -> {remote_file}")

def download_dir(conn, remote_dir, local_dir):
    """Download directory recursively"""
    os.makedirs(local_dir, exist_ok=True)

    # List remote directory
    result = conn.run(f'find {remote_dir} -type f', hide=True)
    files = result.stdout.strip().split('\n')

    for remote_file in files:
        if not remote_file:
            continue
        rel_path = os.path.relpath(remote_file, remote_dir)
        local_file = os.path.join(local_dir, rel_path)

        # Create local directory
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download file
        conn.get(remote_file, local_file)
        print(f"Downloaded: {remote_file} -> {local_file}")

# Usage
conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

upload_dir(conn, './local_data', '/remote/data')
download_dir(conn, '/remote/results', './results')

conn.close()
```

## Error Handling

### Exception Types

```python
from fabric import Connection
from invoke import UnexpectedExit, Failure

conn = Connection(host='host', user='user',
                  connect_kwargs={'password': 'password'})

try:
    result = conn.run('some-command')

except UnexpectedExit as e:
    print(f"Command failed with exit code: {e.result.exited}")
    print(f"Stderr: {e.result.stderr}")

except Failure as e:
    print(f"General failure: {e}")

finally:
    conn.close()
```

### Robust Connection

```python
from fabric import Connection
import time

def robust_connect(host, user, password, max_retries=3, retry_delay=5):
    """Connect with retry logic"""
    for attempt in range(max_retries):
        try:
            conn = Connection(
                host=host, user=user,
                connect_kwargs={'password': password}
            )
            # Test connection
            conn.run('echo ok', hide=True)
            return conn
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise Exception(f"Failed to connect after {max_retries} attempts")

# Usage
conn = robust_connect('192.168.1.100', 'root', 'password')
conn.run('uptime')
conn.close()
```

## Best Practices

1. **Use context managers**
   ```python
   with Connection(host='host', user='user',
                   connect_kwargs={'password': 'pwd'}) as conn:
       conn.run('command')
   ```

2. **Handle errors gracefully**
   ```python
   result = conn.run('command', warn=True, hide=True)
   if result.failed:
       print(f"Error: {result.stderr}")
   ```

3. **Use cd/prefix for clean code**
   ```python
   with conn.cd('/workspace'):
       with conn.prefix('source venv/bin/activate'):
           conn.run('python script.py')
   ```

4. **Hide output for cleaner logs**
   ```python
   result = conn.run('command', hide=True)
   if result.failed:
       print(result.stderr)  # Only show on failure
   ```

5. **Load passwords securely**
   ```python
   import os
   password = os.environ.get('SSH_PASSWORD')
   ```
