#!/usr/bin/env python3
"""
Setup and verify GitCode access token.

Usage:
    python setup_token.py [TOKEN]
    
If TOKEN is not provided, will prompt interactively.
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

API_BASE = "https://api.gitcode.com/api/v5"


def validate_token(token: str) -> tuple[bool, str]:
    """Validate token by making a test API call."""
    url = f"{API_BASE}/user"
    
    headers = {
        "PRIVATE-TOKEN": token,
        "Accept": "application/json",
        "User-Agent": "gitcode-code-reviewer/1.0"
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            user_data = json.loads(response.read().decode("utf-8"))
            return True, user_data.get("login", user_data.get("name", "unknown"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid token"
        return False, f"API error: {e.code}"
    except Exception as e:
        return False, str(e)


def save_to_git_config(token: str) -> bool:
    """Save token to git global config."""
    try:
        subprocess.run(
            ["git", "config", "--global", "gitcode.token", token],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error saving to git config: {e}")
        return False


def get_current_token() -> tuple[str | None, str | None]:
    """Get currently configured token."""
    # Check environment
    env_token = os.environ.get("GITCODE_TOKEN")
    if env_token:
        return env_token, "environment variable GITCODE_TOKEN"
    
    # Check git config
    try:
        result = subprocess.run(
            ["git", "config", "--global", "gitcode.token"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            return result.stdout.strip(), "git config (global)"
    except subprocess.CalledProcessError:
        pass
    
    return None, None


def print_setup_instructions():
    """Print instructions for getting a GitCode token."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                  GitCode Token Setup Guide                     ║
╚════════════════════════════════════════════════════════════════╝

To use the gitcode-code-reviewer skill, you need a GitCode access token.

How to get your token:
1. Login to https://gitcode.com
2. Click your avatar → Settings (设置)
3. Go to "Private Tokens" (私人令牌)
4. Click "Generate Token" (生成令牌)
5. Give it a name like "Code Review"
6. Select scopes:
   ✓ pull_requests (读取和写入)
   ✓ issues (读取和写入)
   ✓ projects (读取)
7. Click "Generate" and copy the token

Token storage options (in priority order):
1. Environment variable: export GITCODE_TOKEN=your_token
2. Git config: git config --global gitcode.token your_token
3. Pass directly: --token your_token (not recommended for regular use)

""")


def main():
    parser = argparse.ArgumentParser(description="Setup GitCode access token")
    parser.add_argument("token", nargs="?", help="GitCode token (optional, will prompt if not provided)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing token")
    parser.add_argument("--save", choices=["env", "git"], default="git", help="Where to save the token")
    
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              GitCode Code Reviewer - Token Setup               ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    # Verify-only mode
    if args.verify_only:
        current_token, source = get_current_token()
        if not current_token:
            print("❌ No token found.")
            print("\nTo set up a token, run without --verify-only:")
            print("  python setup_token.py")
            sys.exit(1)
        
        print(f"Found token from: {source}")
        print("Validating...")
        
        valid, message = validate_token(current_token)
        if valid:
            print(f"✅ Token is valid! Logged in as: @{message}")
            sys.exit(0)
        else:
            print(f"❌ Token validation failed: {message}")
            sys.exit(1)
    
    # Check current token
    current_token, source = get_current_token()
    if current_token:
        print(f"ℹ️  Found existing token from: {source}")
        print("Validating...")
        valid, message = validate_token(current_token)
        if valid:
            print(f"✅ Current token is valid! Logged in as: @{message}")
            response = input("\nDo you want to replace it? [y/N]: ").strip().lower()
            if response != 'y':
                print("Keeping current token.")
                sys.exit(0)
        else:
            print(f"⚠️  Current token is invalid: {message}")
            print("Please enter a new token.\n")
    else:
        print_setup_instructions()
    
    # Get token from user
    if args.token:
        token = args.token
    else:
        print("Please enter your GitCode token:")
        print("(Input will be hidden for security)")
        token = getpass.getpass("Token: ").strip()
    
    if not token:
        print("❌ No token provided.")
        sys.exit(1)
    
    # Validate token
    print("\nValidating token...")
    valid, message = validate_token(token)
    
    if not valid:
        print(f"❌ Token validation failed: {message}")
        print("\nPlease check:")
        print("  - The token is copied correctly")
        print("  - The token has not expired")
        print("  - The token has the required permissions")
        sys.exit(1)
    
    print(f"✅ Token is valid! Logged in as: @{message}")
    
    # Save token
    if args.save == "git":
        print("\nSaving token to git global config...")
        if save_to_git_config(token):
            print("✅ Token saved to git config (--global gitcode.token)")
            print("\nYou can verify it with:")
            print("  git config --global gitcode.token")
        else:
            print("❌ Failed to save token")
            sys.exit(1)
    else:
        print("\nTo use this token, set the environment variable:")
        print(f'  export GITCODE_TOKEN="{token[:10]}..."')
        print("\nOr add it to your shell profile (~/.bashrc, ~/.zshrc, etc.)")
    
    print("\n✨ Setup complete! You can now use the gitcode-code-reviewer skill.")
    print("\nQuick test:")
    print("  python setup_token.py --verify-only")


if __name__ == "__main__":
    main()
