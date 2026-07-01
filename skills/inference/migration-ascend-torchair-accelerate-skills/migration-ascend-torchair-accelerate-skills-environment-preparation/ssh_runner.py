"""
SSH 实时执行器：流式输出 + 即时退出检测 + deadline 兜底
解决 paramiko get_pty=True 下 exit_status 不触发导致长时间空等问题。

用法:
    from ssh_runner import run_remote
    run_remote("docker exec xxx python3 test.py", host="***.***.*.*", 
               user="****", password="**********", deadline=600)
"""

import paramiko
import time
import socket
import threading
import sys


class RemoteRunner:
    """在远端执行命令，实时流式输出 stdout/stderr，检测进程退出后立即返回"""

    def __init__(self, host, port=22, user="****", password=None, key_file=None):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.key_file = key_file

    def _connect(self):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(hostname=self.host, port=self.port, username=self.user)
        if self.password:
            kwargs["password"] = self.password
        if self.key_file:
            kwargs["key_filename"] = self.key_file
        c.connect(**kwargs, timeout=15)
        return c

    def run(self, cmd, deadline=600, get_pty=True, print_output=True, workdir=None):
        """
        执行命令，实时流式输出，进程结束或报错立即返回。

        Args:
            cmd: 要执行的命令
            deadline: 总超时秒数（整个命令允许的最长运行时间）
            get_pty: 是否分配伪终端（需要实时输出时设 True）
            print_output: True=打印到 stdout, False=仅收集
            workdir: 远端工作目录

        Returns:
            dict: {stdout, stderr, exit_code}
        """
        if workdir:
            cmd = f"cd {workdir} && {cmd}"

        client = self._connect()
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=get_pty)

        # 分别收集
        out_buf = []
        err_buf = []

        # 短 per-read 超时——读完就检查退出状态
        stdout.channel.settimeout(2)
        stderr.channel.settimeout(2)

        start = time.time()

        while True:
            # 检查进程是否已退出
            exited = stdout.channel.exit_status_ready()

            # 读取可用数据
            try:
                if stdout.channel.recv_ready():
                    chunk = stdout.channel.recv(8192)
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        out_buf.append(text)
                        if print_output:
                            sys.stdout.write(text)
                            sys.stdout.flush()
            except socket.timeout:
                pass

            try:
                if stderr.channel.recv_stderr_ready():
                    chunk = stderr.channel.recv_stderr(8192)
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        err_buf.append(text)
                        if print_output:
                            sys.stderr.write(text)
                            sys.stderr.flush()
            except socket.timeout:
                pass

            # 进程已退出 → 读完剩余数据后立即返回
            if exited:
                # 排空残留
                for _ in range(50):  # 最多再试 50 轮
                    try:
                        if stdout.channel.recv_ready():
                            chunk = stdout.channel.recv(8192)
                            if chunk:
                                text = chunk.decode("utf-8", errors="replace")
                                out_buf.append(text)
                                if print_output:
                                    sys.stdout.write(text)
                                    sys.stdout.flush()
                        elif stderr.channel.recv_stderr_ready():
                            chunk = stderr.channel.recv_stderr(8192)
                            if chunk:
                                text = chunk.decode("utf-8", errors="replace")
                                err_buf.append(text)
                                if print_output:
                                    sys.stderr.write(text)
                                    sys.stderr.flush()
                        else:
                            break
                    except socket.timeout:
                        break
                break

            # deadline 兜底
            if time.time() - start > deadline:
                sys.stderr.write(f"\n[RemoteRunner] deadline {deadline}s reached, closing\n")
                stdout.channel.close()
                break

            # 避免空转——很短 sleep 后继续轮询
            time.sleep(0.1)

        exit_code = stdout.channel.recv_exit_status() if stdout.channel.exit_status_ready() else -1
        client.close()

        return {
            "stdout": "".join(out_buf),
            "stderr": "".join(err_buf),
            "exit_code": exit_code,
        }


# ====== 便捷函数 ======

_default_runner = None


def get_runner(host="***.***.*.*", user="****", password="**********"):
    global _default_runner
    if _default_runner is None or _default_runner.host != host:
        _default_runner = RemoteRunner(host=host, user=user, password=password)
    return _default_runner


def run_remote(cmd, deadline=600, host="***.***.*.*", user="****", password="**********",
               workdir=None, print_output=True):
    """一行调用"""
    runner = RemoteRunner(host=host, user=user, password=password)
    return runner.run(cmd, deadline=deadline, print_output=print_output, workdir=workdir)


# ====== 自测 ======
if __name__ == "__main__":
    import sys
    deadline = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    host = sys.argv[2] if len(sys.argv) > 2 else "***.***.*.*"
    user = sys.argv[3] if len(sys.argv) > 3 else "****"
    pwd = sys.argv[4] if len(sys.argv) > 4 else "**********"

    r = run_remote("echo hello; sleep 2; echo world; npu-smi info 2>&1 | head -5",
                   deadline=deadline, host=host, user=user, password=pwd)
    print("\n--- RESULT ---")
    print(f"exit_code={r['exit_code']}")
    print(f"stdout len={len(r['stdout'])}")
    print(f"stderr len={len(r['stderr'])}")
