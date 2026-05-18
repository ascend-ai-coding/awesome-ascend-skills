# Paramiko Advanced Operations

## Docker Management

### Docker Management Class

```python
import paramiko
import os

class RemoteDocker:
    def __init__(self, host, port, user, password):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=host, port=port, username=user, password=password)

    def exec(self, command, timeout=60):
        """Execute command and return result"""
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        return {
            'stdout': stdout.read().decode(),
            'stderr': stderr.read().decode(),
            'exit_code': stdout.channel.recv_exit_status()
        }

    def list_containers(self, all=False):
        """List containers"""
        flag = '-a' if all else ''
        result = self.exec(f'docker ps {flag} --format "{{{{.Names}}}} {{{{.Image}}}} {{{{.Status}}}}"')
        if result['exit_code'] == 0:
            return [line.split() for line in result['stdout'].strip().split('\n') if line]
        return []

    def create_container(self, name, image, **kwargs):
        """Create and start container"""
        cmd = f"docker run -d --name {name}"

        if kwargs.get('privileged'):
            cmd += " --privileged"

        if kwargs.get('network'):
            cmd += f" --network {kwargs['network']}"

        if kwargs.get('memory'):
            cmd += f" --shm-size={kwargs['memory']}"

        if kwargs.get('mounts'):
            for src, dst in kwargs['mounts'].items():
                cmd += f" -v {src}:{dst}"

        if kwargs.get('env'):
            for key, value in kwargs['env'].items():
                cmd += f" -e {key}={value}"

        if kwargs.get('ports'):
            for host_port, container_port in kwargs['ports'].items():
                cmd += f" -p {host_port}:{container_port}"

        cmd += f" {image}"

        if kwargs.get('command'):
            cmd += f" {kwargs['command']}"

        return self.exec(cmd)

    def start_container(self, name):
        """Start container"""
        return self.exec(f'docker start {name}')

    def stop_container(self, name):
        """Stop container"""
        return self.exec(f'docker stop {name}')

    def remove_container(self, name, force=False):
        """Remove container"""
        flag = '-f' if force else ''
        return self.exec(f'docker rm {flag} {name}')

    def exec_in_container(self, container, command):
        """Execute command in container"""
        return self.exec(f'docker exec {container} {command}')

    def exec_in_container_interactive(self, container, command):
        """Execute interactive command in container"""
        return self.exec(f'docker exec -it {container} {command}')

    def copy_to_container(self, local_path, container, remote_path):
        """Copy file from local to container"""
        # Upload to host temp
        sftp = self.client.open_sftp()
        temp_name = os.path.basename(local_path)
        temp_path = f'/tmp/{temp_name}'
        sftp.put(local_path, temp_path)
        sftp.close()

        # Copy to container
        result = self.exec(f'docker cp {temp_path} {container}:{remote_path}')

        # Clean up
        self.exec(f'rm {temp_path}')

        return result

    def copy_from_container(self, container, remote_path, local_path):
        """Copy file from container to local"""
        # Copy from container to host
        temp_name = os.path.basename(remote_path)
        temp_path = f'/tmp/{temp_name}'
        self.exec(f'docker cp {container}:{remote_path} {temp_path}')

        # Download from host
        sftp = self.client.open_sftp()
        sftp.get(temp_path, local_path)
        sftp.close()

        # Clean up
        self.exec(f'rm {temp_path}')

    def get_container_logs(self, name, tail=100):
        """Get container logs"""
        return self.exec(f'docker logs --tail {tail} {name}')

    def follow_container_logs(self, name, callback):
        """Follow container logs in real-time"""
        transport = self.client.get_transport()
        channel = transport.open_session()
        channel.exec_command(f'docker logs -f {name}')

        while True:
            if channel.recv_ready():
                line = channel.recv(1024).decode()
                callback(line)
            if channel.exit_status_ready():
                break

    def close(self):
        """Close connection"""
        self.client.close()

# Usage
docker = RemoteDocker('192.168.1.100', 22, 'root', 'password')

# List containers
for container in docker.list_containers(all=True):
    print(container)

# Create container
docker.create_container(
    name='my-app',
    image='python:3.11',
    network='host',
    mounts={
        '/home/data': '/data',
        '/home/models': '/models'
    },
    env={
        'APP_ENV': 'production'
    }
)

# Copy script and execute
docker.copy_to_container('./app.py', 'my-app', '/workspace/')
result = docker.exec_in_container('my-app', 'python /workspace/app.py')
print(result['stdout'])

# Get results
docker.copy_from_container('my-app', '/workspace/output.csv', './output.csv')

# Cleanup
docker.remove_container('my-app', force=True)
docker.close()
```

## Jump Host

### Using ProxyCommand

```python
import paramiko
from paramiko import SSHClient, ProxyCommand

# ProxyCommand requires ssh on local machine
proxy_cmd = 'ssh -W 192.168.1.100:22 jump-user@jump.example.com'
sock = paramiko.ProxyCommand(proxy_cmd)

client = SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect through jump host (no jump host password needed if using keys)
client.connect('192.168.1.100', username='root', password='target_password', sock=sock)

stdin, stdout, stderr = client.exec_command('hostname')
print(stdout.read().decode())

client.close()
```

### Nested SSH (Pure Python)

```python
import paramiko
from paramiko import SSHClient

# Connect to jump host first
jump_client = SSHClient()
jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
jump_client.connect(
    hostname='jump.example.com',
    username='jump-user',
    password='jump-password'
)

# Create tunnel through jump host
transport = jump_client.get_transport()
channel = transport.open_channel(
    'direct-tcpip',
    ('192.168.1.100', 22),  # Target host
    ('', 0)                  # Local endpoint
)

# Connect to target through tunnel
target_client = SSHClient()
target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
target_client.connect(
    hostname='192.168.1.100',
    username='root',
    password='target-password',
    sock=channel
)

# Execute on target
stdin, stdout, stderr = target_client.exec_command('hostname')
print(stdout.read().decode())

# Close connections
target_client.close()
jump_client.close()
```

### Jump Host Helper Class

```python
import paramiko

class JumpHostSSH:
    def __init__(self, jump_host, jump_user, jump_password, target_host, target_user, target_password):
        # Connect to jump host
        self.jump_client = paramiko.SSHClient()
        self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.jump_client.connect(
            hostname=jump_host,
            username=jump_user,
            password=jump_password
        )

        # Create tunnel
        transport = self.jump_client.get_transport()
        channel = transport.open_channel(
            'direct-tcpip',
            (target_host, 22),
            ('', 0)
        )

        # Connect to target
        self.target_client = paramiko.SSHClient()
        self.target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.target_client.connect(
            hostname=target_host,
            username=target_user,
            password=target_password,
            sock=channel
        )

    def exec_command(self, command, timeout=60):
        """Execute command on target"""
        stdin, stdout, stderr = self.target_client.exec_command(command, timeout=timeout)
        return {
            'stdout': stdout.read().decode(),
            'stderr': stderr.read().decode(),
            'exit_code': stdout.channel.recv_exit_status()
        }

    def close(self):
        """Close all connections"""
        self.target_client.close()
        self.jump_client.close()

# Usage
ssh = JumpHostSSH(
    jump_host='jump.example.com',
    jump_user='jump-user',
    jump_password='jump-pass',
    target_host='192.168.1.100',
    target_user='root',
    target_password='target-pass'
)

result = ssh.exec_command('hostname && uptime')
print(result['stdout'])
ssh.close()
```

## Complete Example: Remote Service Deployment

```python
#!/usr/bin/env python3
"""
Deploy a service on a remote host using paramiko
"""

import paramiko
import os
import time

class RemoteDeployer:
    def __init__(self, host, user, password):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=host, username=user, password=password)
        self.sftp = self.client.open_sftp()

    def exec(self, command):
        stdin, stdout, stderr = self.client.exec_command(command)
        return stdout.read().decode(), stderr.read().decode()

    def check_environment(self):
        """Check server status"""
        print("Checking environment...")

        # Check system
        stdout, _ = self.exec('uptime')
        print(f"Uptime: {stdout.strip()}")

        # Check Docker
        stdout, _ = self.exec('docker --version')
        print(f"Docker: {stdout.strip()}")

    def create_container(self, name, image, **kwargs):
        """Create service container"""
        print(f"Creating container {name}...")

        cmd = f"docker run -d --name {name}"

        if kwargs.get('network', 'host'):
            cmd += f" --network {kwargs.get('network', 'host')}"

        if kwargs.get('mounts'):
            for src, dst in kwargs['mounts'].items():
                cmd += f" -v {src}:{dst}"

        if kwargs.get('env'):
            for key, value in kwargs['env'].items():
                cmd += f" -e {key}={value}"

        if kwargs.get('ports'):
            for hp, cp in kwargs['ports'].items():
                cmd += f" -p {hp}:{cp}"

        cmd += f" {image}"

        if kwargs.get('command'):
            cmd += f" {kwargs['command']}"

        stdout, stderr = self.exec(cmd)
        if stderr:
            print(f"Error: {stderr}")
        else:
            print(f"Container created: {stdout.strip()}")

    def deploy_service(self, container, start_script_content, script_name='start.sh'):
        """Deploy service using a start script"""
        print("Deploying service...")

        local_script = f'/tmp/{script_name}'
        remote_script = f'/tmp/{script_name}'

        with open(local_script, 'w') as f:
            f.write(start_script_content)

        self.sftp.put(local_script, remote_script)
        os.remove(local_script)

        # Copy to container and execute
        self.exec(f'docker cp {remote_script} {container}:/workspace/')
        self.exec(f'docker exec {container} chmod +x /workspace/{script_name}')
        self.exec(f'docker exec -d {container} bash /workspace/{script_name}')

        print("Service deployment started")

    def wait_for_service(self, container, port=8000, timeout=300):
        """Wait for service to be ready"""
        print("Waiting for service...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            stdout, _ = self.exec(f'docker exec {container} curl -s localhost:{port}/health')
            if 'ok' in stdout.lower():
                print("Service is ready!")
                return True
            time.sleep(5)

        print("Timeout waiting for service")
        return False

    def test_service(self, container, port=8000, endpoint='/'):
        """Test service endpoint"""
        print("Testing service...")

        cmd = f'docker exec {container} curl -s localhost:{port}{endpoint}'
        stdout, _ = self.exec(cmd)
        print(f"Response: {stdout[:200]}")

    def close(self):
        self.sftp.close()
        self.client.close()

# Usage
if __name__ == '__main__':
    deployer = RemoteDeployer('192.168.1.100', 'root', 'password')

    deployer.check_environment()
    deployer.create_container(
        'my-service',
        'python:3.11',
        network='host',
        mounts={'/home/data': '/data'}
    )

    start_script = """#!/bin/bash
cd /workspace
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8000
"""
    deployer.deploy_service('my-service', start_script)

    if deployer.wait_for_service('my-service'):
        deployer.test_service('my-service')

    deployer.close()
```
