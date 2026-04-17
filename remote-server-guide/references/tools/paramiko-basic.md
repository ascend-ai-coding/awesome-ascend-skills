# Paramiko Basic Operations

## Overview

Paramiko is a pure-Python implementation of SSHv2 protocol. It provides full control over SSH operations and is ideal for complex workflows and Python applications.

## Installation

```bash
# pip (recommended)
pip install paramiko

# pip with specific version
pip install paramiko==3.4.0

# conda
conda install -c conda-forge paramiko

# Verify installation
python3 -c "import paramiko; print(paramiko.__version__)"
```

## Connection

### Simple Connection

```python
import paramiko

# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect
client.connect(
    hostname='192.168.1.100',
    port=22,
    username='root',
    password='your_password'
)

# Use connection...

# Close
client.close()
```

### Context Manager (Recommended)

```python
import paramiko

# Using context manager for auto-cleanup
with paramiko.SSHClient() as client:
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname='host', username='user', password='password')
    # Connection auto-closes when exiting context
```

### Connection with All Options

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect(
    hostname='192.168.1.100',
    port=22,
    username='root',
    password='password',
    timeout=30,              # Connection timeout
    banner_timeout=30,       # Banner timeout
    auth_timeout=30,         # Authentication timeout
    look_for_keys=False,     # Don't look for SSH keys
    allow_agent=False,       # Don't use SSH agent
)
```

### SSH Key Authentication

```python
import paramiko
from paramiko import RSAKey

# Load private key
private_key = RSAKey.from_private_key_file('/path/to/private_key')

# Or with passphrase
private_key = RSAKey.from_private_key_file(
    '/path/to/private_key',
    password='key_passphrase'
)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    hostname='host',
    username='user',
    pkey=private_key
)
```

## Remote Command Execution

### Basic Execution

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='host', username='user', password='password')

# Execute command
stdin, stdout, stderr = client.exec_command('ls -la')

# Read output
print(stdout.read().decode())
print(stderr.read().decode())

# Get exit code
exit_code = stdout.channel.recv_exit_status()
print(f"Exit code: {exit_code}")

client.close()
```

### Wrapper Function

```python
import paramiko

def ssh_exec(host, port, user, password, command, timeout=30):
    """Execute SSH command and return result dict"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=timeout
        )

        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        return {
            'success': True,
            'stdout': stdout.read().decode(),
            'stderr': stderr.read().decode(),
            'exit_code': stdout.channel.recv_exit_status()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        client.close()

# Usage
result = ssh_exec('192.168.1.100', 22, 'root', 'password', 'uptime')
if result['success']:
    print(result['stdout'])
else:
    print(f"Error: {result['error']}")
```

### Multiple Commands

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='host', username='user', password='password')

commands = [
    'cd /workspace',
    'ls -la',
    'python script.py'
]

for cmd in commands:
    stdin, stdout, stderr = client.exec_command(cmd)
    output = stdout.read().decode()
    error = stderr.read().decode()
    exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {error}")
        break

    print(output)

client.close()
```

### Long-Running Commands

```python
import paramiko
import time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='host', username='user', password='password')

# Execute long-running command
transport = client.get_transport()
channel = transport.open_session()
channel.exec_command('python long_training.py')

# Stream output in real-time
while not channel.exit_status_ready():
    if channel.recv_ready():
        print(channel.recv(1024).decode(), end='')
    if channel.recv_stderr_ready():
        print(channel.recv_stderr(1024).decode(), end='')
    time.sleep(0.1)

print(f"\nExit code: {channel.recv_exit_status()}")
client.close()
```

## File Transfer (SFTP)

### Basic SFTP Operations

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='host', username='user', password='password')

# Open SFTP session
sftp = client.open_sftp()

# Upload file
sftp.put('/local/file.txt', '/remote/path/file.txt')

# Download file
sftp.get('/remote/file.txt', '/local/file.txt')

# Close SFTP
sftp.close()
client.close()
```

### SFTP Wrapper Class

```python
import paramiko
import os

class SFTPClient:
    def __init__(self, host, port, user, password):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=host, port=port, username=user, password=password)
        self.sftp = self.client.open_sftp()

    def upload(self, local_path, remote_path):
        """Upload file to remote"""
        self.sftp.put(local_path, remote_path)

    def download(self, remote_path, local_path):
        """Download file from remote"""
        self.sftp.get(remote_path, local_path)

    def upload_dir(self, local_dir, remote_dir):
        """Upload directory recursively"""
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                remote_path = os.path.join(remote_dir, rel_path)

                # Create remote directory if needed
                remote_dir_path = os.path.dirname(remote_path)
                try:
                    self.sftp.stat(remote_dir_path)
                except FileNotFoundError:
                    self._mkdir_p(remote_dir_path)

                self.sftp.put(local_path, remote_path)

    def download_dir(self, remote_dir, local_dir):
        """Download directory recursively"""
        os.makedirs(local_dir, exist_ok=True)

        for entry in self.sftp.listdir_attr(remote_dir):
            remote_path = os.path.join(remote_dir, entry.filename)
            local_path = os.path.join(local_dir, entry.filename)

            if entry.st_mode & 0o170000 == 0o040000:  # Directory
                self.download_dir(remote_path, local_path)
            else:  # File
                self.sftp.get(remote_path, local_path)

    def _mkdir_p(self, path):
        """Create directory recursively"""
        dirs = path.split('/')
        current = ''
        for d in dirs:
            current += '/' + d
            try:
                self.sftp.stat(current)
            except FileNotFoundError:
                self.sftp.mkdir(current)

    def list_dir(self, remote_path):
        """List directory contents"""
        return self.sftp.listdir(remote_path)

    def remove(self, remote_path):
        """Remove file"""
        self.sftp.remove(remote_path)

    def close(self):
        """Close connections"""
        self.sftp.close()
        self.client.close()

# Usage
sftp = SFTPClient('192.168.1.100', 22, 'root', 'password')
sftp.upload_dir('./local_data', '/remote/data')
sftp.download_dir('/remote/results', './results')
sftp.close()
```

### Progress Callback

```python
import paramiko
import os

def progress_callback(transferred, total):
    percent = (transferred / total) * 100
    print(f"\rProgress: {percent:.1f}%", end='')

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='host', username='user', password='password')

sftp = client.open_sftp()

# Upload with progress
file_size = os.path.getsize('/local/large_file.bin')
sftp.put('/local/large_file.bin', '/remote/large_file.bin', callback=progress_callback)
print()  # New line after progress

sftp.close()
client.close()
```

## Error Handling

### Exception Types

```python
import paramiko
from paramiko import AuthenticationException, SSHException, BadHostKeyException

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname='host', username='user', password='password')

except AuthenticationException:
    print("Authentication failed - check username/password")
except BadHostKeyException:
    print("Host key verification failed")
except SSHException as e:
    print(f"SSH error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    client.close()
```

### Robust Connection Function

```python
import paramiko
import time

def robust_connect(host, user, password, max_retries=3, retry_delay=5):
    """Connect with retry logic"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(max_retries):
        try:
            client.connect(
                hostname=host,
                username=user,
                password=password,
                timeout=30
            )
            return client
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise Exception(f"Failed to connect after {max_retries} attempts")

# Usage
try:
    client = robust_connect('192.168.1.100', 'root', 'password')
    # Use client...
    client.close()
except Exception as e:
    print(f"Connection failed: {e}")
```

## Best Practices

1. **Always use context managers or try/finally**
   ```python
   # Good
   with paramiko.SSHClient() as client:
       client.connect(...)
       # Use client
   # Auto-closed

   # Or
   client = paramiko.SSHClient()
   try:
       client.connect(...)
   finally:
       client.close()
   ```

2. **Handle exceptions properly**
   ```python
   from paramiko import AuthenticationException, SSHException

   try:
       client.connect(...)
   except AuthenticationException:
       # Handle auth error
   except SSHException:
       # Handle SSH error
   ```

3. **Use timeout values**
   ```python
   client.connect(..., timeout=30)
   stdin, stdout, stderr = client.exec_command(..., timeout=60)
   ```

4. **Load passwords from secure sources**
   ```python
   import os
   password = os.environ.get('SSH_PASSWORD')
   # Or from a config file / secrets manager
   ```
