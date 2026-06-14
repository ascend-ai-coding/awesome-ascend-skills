#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def read_file(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return f.read()


def write_file(file_path: Path, content: str) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write {file_path}: {e}")
        return False


def read_json(file_path: Path) -> Optional[Dict]:
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None


def write_json(file_path: Path, data: Dict) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON {file_path}: {e}")
        return False


def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def detect_package_manager() -> Optional[str]:
    if os.path.exists("/bin/rpm") or os.path.exists("/usr/bin/rpm"):
        return "rpm"
    if os.path.exists("/bin/dpkg") or os.path.exists("/usr/bin/dpkg"):
        return "deb"
    if os.path.exists("/bin/pacman") or os.path.exists("/usr/bin/pacman"):
        return "pacman"
    return None


def get_os_release_info() -> Dict[str, str]:
    os_info = {}
    os_release_path = "/etc/os-release"
    
    if os.path.exists(os_release_path):
        with open(os_release_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"\'')
                    os_info[key] = value
    
    return os_info


def get_architecture() -> str:
    returncode, stdout, _ = run_command(['uname', '-m'])
    if returncode == 0:
        return stdout.strip()
    return "unknown"


def normalize_os_name(name: str, version: str, arch: str) -> str:
    name_clean = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_')
    version_clean = re.sub(r'[^a-zA-Z0-9.]+', '_', version).strip('_')
    return f"{name_clean}_{version_clean}_{arch}"


def generate_constant_name(os_name: str) -> str:
    return os_name.upper().replace('.', '_').replace('-', '_')


def parse_installed_packages_rpm() -> List[Dict]:
    packages = []
    returncode, stdout, _ = run_command(['rpm', '-qa', '--queryformat', '%{NAME} %{VERSION}-%{RELEASE} %{ARCH}\n'])
    
    if returncode == 0:
        for line in stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    packages.append({
                        'name': parts[0],
                        'version': parts[1],
                        'arch': parts[2] if len(parts) > 2 else ''
                    })
    
    return packages


def parse_installed_packages_deb() -> List[Dict]:
    packages = []
    returncode, stdout, _ = run_command(['dpkg', '-l'])
    
    if returncode == 0:
        lines = stdout.strip().split('\n')
        for line in lines[5:]:
            parts = line.split()
            if len(parts) >= 3:
                packages.append({
                    'name': parts[1],
                    'version': parts[2],
                    'arch': parts[3] if len(parts) > 3 else ''
                })
    
    return packages


def get_installed_packages() -> List[Dict]:
    pkg_manager = detect_package_manager()
    
    if pkg_manager == "rpm":
        return parse_installed_packages_rpm()
    elif pkg_manager == "deb":
        return parse_installed_packages_deb()
    
    return []


def get_rpm_repos() -> List[Dict]:
    repos = []
    yum_repos_dir = Path("/etc/yum.repos.d")
    
    if yum_repos_dir.exists():
        for repo_file in yum_repos_dir.glob("*.repo"):
            content = read_file(repo_file)
            if content:
                repos.append({
                    'type': 'rpm',
                    'file': str(repo_file),
                    'content': content
                })
    
    return repos


def get_deb_repos() -> List[Dict]:
    repos = []
    apt_sources_file = Path("/etc/apt/sources.list")
    apt_sources_dir = Path("/etc/apt/sources.list.d")
    
    if apt_sources_file.exists():
        content = read_file(apt_sources_file)
        if content:
            repos.append({
                'type': 'deb',
                'file': str(apt_sources_file),
                'content': content
            })
    
    if apt_sources_dir.exists():
        for source_file in apt_sources_dir.glob("*.list"):
            content = read_file(source_file)
            if content:
                repos.append({
                    'type': 'deb',
                    'file': str(source_file),
                    'content': content
                })
    
    return repos


def get_system_repos() -> List[Dict]:
    pkg_manager = detect_package_manager()
    
    if pkg_manager == "rpm":
        return get_rpm_repos()
    elif pkg_manager == "deb":
        return get_deb_repos()
    
    return []


def check_duplicate_in_file(file_path: Path, pattern: str) -> bool:
    content = read_file(file_path)
    if content and pattern in content:
        return True
    return False


def check_duplicate_in_json_list(file_path: Path, item: str, key: str = None) -> bool:
    data = read_json(file_path)
    if not data:
        return False
    
    if key and key in data:
        return item in data[key]
    elif isinstance(data, list):
        return item in data
    
    return False


def append_to_python_class(file_path: Path, class_name: str, line: str) -> bool:
    content = read_file(file_path)
    if not content:
        return False
    
    class_pattern = f'class {class_name}'
    if class_pattern not in content:
        return False
    
    class_start = content.find(class_pattern)
    insert_pos = content.find('\n', class_start) + 1
    
    new_content = content[:insert_pos] + line + '\n' + content[insert_pos:]
    return write_file(file_path, new_content)


def append_to_python_list(file_path: Path, list_name: str, item: str) -> bool:
    content = read_file(file_path)
    if not content:
        return False
    
    list_pattern = f'{list_name} = ['
    if list_pattern not in content:
        return False
    
    list_start = content.find(list_pattern)
    list_end = content.find(']', list_start)
    
    if list_end == -1:
        return False
    
    list_content = content[list_start:list_end + 1]
    
    if item in list_content:
        return True
    
    new_list_content = list_content.replace(']', f',\n        "{item}"\n    ]')
    new_content = content[:list_start] + new_list_content + content[list_end + 1:]
    
    return write_file(file_path, new_content)


def append_to_python_dict(file_path: Path, dict_name: str, key: str, value: str) -> bool:
    content = read_file(file_path)
    if not content:
        return False
    
    dict_pattern = f'{dict_name} = {{'
    if dict_pattern not in content:
        return False
    
    dict_start = content.find(dict_pattern)
    dict_end = content.find('}', dict_start)
    
    if dict_end == -1:
        return False
    
    dict_content = content[dict_start:dict_end + 1]
    
    if key in dict_content:
        return True
    
    entry = f'        "{key}": {value},\n'
    new_content = content[:dict_end] + entry + content[dict_end:]
    
    return write_file(file_path, new_content)


def validate_os_name(os_name: str) -> bool:
    pattern = r'^[a-zA-Z][a-zA-Z0-9_.-]*$'
    return bool(re.match(pattern, os_name))


def validate_hardware_list(hardware_list: List[str]) -> bool:
    valid_hardware = {'I2', 'A2', 'A3', 'A5', 'A7', 'A9'}
    return all(h in valid_hardware for h in hardware_list)
