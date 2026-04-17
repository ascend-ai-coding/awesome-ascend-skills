# Fabric Advanced Operations

## Docker Management

### Docker Management Class

```python
from fabric import Connection
import os

class RemoteDocker:
    def __init__(self, host, user, password):
        self.conn = Connection(
            host=host, user=user,
            connect_kwargs={'password': password}
        )

    def list_containers(self, all=False):
        """List containers"""
        flag = '-a' if all else ''
        result = self.conn.run(
            f'docker ps {flag} --format "{{{{.Names}}}} {{{{.Image}}}} {{{{.Status}}}}"',
            hide=True
        )
        if result.ok:
            return [line.split() for line in result.stdout.strip().split('\n') if line]
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

        return self.conn.run(cmd, hide=True)

    def start_container(self, name):
        """Start container"""
        return self.conn.run(f'docker start {name}', hide=True)

    def stop_container(self, name):
        """Stop container"""
        return self.conn.run(f'docker stop {name}', hide=True)

    def remove_container(self, name, force=False):
        """Remove container"""
        flag = '-f' if force else ''
        return self.conn.run(f'docker rm {flag} {name}', hide=True, warn=True)

    def exec(self, container, command):
        """Execute command in container"""
        return self.conn.run(f'docker exec {container} {command}', hide=True)

    def exec_interactive(self, container, command):
        """Execute interactive command"""
        return self.conn.run(f'docker exec -it {container} {command}', pty=True)

    def copy_to(self, local_path, container, remote_path):
        """Copy file from local to container"""
        # Upload to host temp
        temp_name = os.path.basename(local_path)
        temp_path = f'/tmp/{temp_name}'
        self.conn.put(local_path, temp_path)

        # Copy to container
        self.conn.run(f'docker cp {temp_path} {container}:{remote_path}', hide=True)

        # Clean up
        self.conn.run(f'rm {temp_path}', hide=True)

    def copy_from(self, container, remote_path, local_path):
        """Copy file from container to local"""
        # Copy from container to host
        temp_name = os.path.basename(remote_path)
        temp_path = f'/tmp/{temp_name}'
        self.conn.run(f'docker cp {container}:{remote_path} {temp_path}', hide=True)

        # Download from host
        self.conn.get(temp_path, local_path)

        # Clean up
        self.conn.run(f'rm {temp_path}', hide=True)

    def logs(self, name, tail=100, follow=False):
        """Get container logs"""
        cmd = f'docker logs --tail {tail}'
        if follow:
            cmd += ' -f'
        return self.conn.run(f'{cmd} {name}', hide=not follow)

    def close(self):
        """Close connection"""
        self.conn.close()

# Usage
docker = RemoteDocker('192.168.1.100', 'root', 'password')

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

# Copy and execute
docker.copy_to('./app.py', 'my-app', '/workspace/')
result = docker.exec('my-app', 'python /workspace/app.py')
print(result.stdout)

# Get results
docker.copy_from('my-app', '/workspace/output.csv', './output.csv')

# Cleanup
docker.remove_container('my-app', force=True)
docker.close()
```

## Jump Host (Gateway)

### Basic Gateway

```python
from fabric import Connection

# Define jump host
jump = Connection(
    host='jump.example.com',
    user='jump-user',
    connect_kwargs={'password': 'jump-password'}
)

# Connect via gateway
target = Connection(
    host='192.168.1.100',
    user='root',
    gateway=jump,
    connect_kwargs={'password': 'target-password'}
)

# Execute on target
result = target.run('hostname', hide=True)
print(result.stdout)

# Close connections
target.close()
jump.close()
```

### Gateway Helper Function

```python
from fabric import Connection

def create_gateway_connection(jump_config, target_config):
    """Create connection through jump host"""
    jump = Connection(
        host=jump_config['host'],
        user=jump_config['user'],
        connect_kwargs={'password': jump_config['password']}
    )

    target = Connection(
        host=target_config['host'],
        user=target_config['user'],
        gateway=jump,
        connect_kwargs={'password': target_config['password']}
    )

    return target, jump

# Usage
target, jump = create_gateway_connection(
    jump_config={
        'host': 'jump.example.com',
        'user': 'jump-user',
        'password': 'jump-pass'
    },
    target_config={
        'host': '192.168.1.100',
        'user': 'root',
        'password': 'target-pass'
    }
)

result = target.run('hostname && uptime', hide=True)
print(result.stdout)

target.close()
jump.close()
```

### Multiple Jump Hosts

```python
from fabric import Connection

# Jump host 1
jump1 = Connection(
    host='jump1.example.com',
    user='user1',
    connect_kwargs={'password': 'pass1'}
)

# Jump host 2 (through jump1)
jump2 = Connection(
    host='jump2.example.com',
    user='user2',
    gateway=jump1,
    connect_kwargs={'password': 'pass2'}
)

# Target (through jump2)
target = Connection(
    host='192.168.1.100',
    user='root',
    gateway=jump2,
    connect_kwargs={'password': 'target-pass'}
)

result = target.run('hostname', hide=True)
print(result.stdout)

target.close()
jump2.close()
jump1.close()
```

## Parallel Execution

### Multiple Hosts

```python
from fabric import Connection
from concurrent.futures import ThreadPoolExecutor

def execute_on_host(host, user, password, command):
    """Execute command on a single host"""
    conn = Connection(
        host=host, user=user,
        connect_kwargs={'password': password}
    )
    result = conn.run(command, hide=True)
    conn.close()
    return host, result.stdout, result.ok

# Parallel execution
hosts = ['192.168.1.100', '192.168.1.101', '192.168.1.102']
user = 'root'
password = 'password'
command = 'uptime && df -h / | tail -1'

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(execute_on_host, host, user, password, command)
        for host in hosts
    ]

    for future in futures:
        host, stdout, ok = future.result()
        print(f"=== {host} ===")
        if ok:
            print(stdout)
        else:
            print("Failed")
```

### Using Fabric's Group

```python
from fabric import SerialGroup, ThreadingGroup

# Serial execution (one by one)
hosts = SerialGroup(
    'root@192.168.1.100',
    'root@192.168.1.101',
    'root@192.168.1.102',
    connect_kwargs={'password': 'password'}
)

# Execute on all hosts
results = hosts.run('uptime', hide=True)

for connection, result in results.items():
    print(f"=== {connection.host} ===")
    print(result.stdout)

# Parallel execution
hosts = ThreadingGroup(
    'root@192.168.1.100',
    'root@192.168.1.101',
    'root@192.168.1.102',
    connect_kwargs={'password': 'password'}
)

results = hosts.run('docker ps', hide=True)
```

## Task-Based CLI

### tasks.py

```python
# tasks.py - Define tasks for fab CLI
from fabric import task, Connection

@task
def deploy(c, host='192.168.1.100', image='my-image:latest'):
    """Deploy service"""
    conn = Connection(
        host=host,
        user='root',
        connect_kwargs={'password': c.connection_password}
    )

    # Create container
    conn.run(f'docker run -d --name my-service --network host {image}')

    conn.close()

@task
def status(c, host='192.168.1.100'):
    """Check service status"""
    conn = Connection(
        host=host,
        user='root',
        connect_kwargs={'password': c.connection_password}
    )

    conn.run('docker ps | grep my-service')
    conn.run('curl -s localhost:8000/health')

    conn.close()

@task
def logs(c, host='192.168.1.100', tail=100):
    """View service logs"""
    conn = Connection(
        host=host,
        user='root',
        connect_kwargs={'password': c.connection_password}
    )

    conn.run(f'docker logs --tail {tail} my-service')

    conn.close()
```

### Running Tasks

```bash
# List available tasks
fab --list

# Run task
fab -p password deploy --host=192.168.1.100

# Run with image parameter
fab -p password deploy --host=192.168.1.100 --image=my-image:v2

# Check status
fab -p password status --host=192.168.1.100

# View logs
fab -p password logs --host=192.168.1.100 --tail=50
```

## Complete Example: Service Deployment

Key steps for deploying a service with Fabric:

1. Create RemoteDocker instance with connection
2. Create container with mounts and environment
3. Upload and execute start script
4. Wait for health check
5. Test service endpoint

```python
# Quick pattern
docker = RemoteDocker('192.168.1.100', 'root', 'password')
docker.create_container(
    name='my-service',
    image='python:3.11',
    network='host',
    mounts={'/home/data': '/data'},
    env={'APP_ENV': 'production'}
)
docker.exec('my-service', 'python /workspace/server.py --port 8000')
# Check health and run tests...
docker.close()
```
