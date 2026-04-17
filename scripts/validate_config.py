#!/usr/bin/env python3
"""Validate external-sources.yml configuration file."""

import sys
import yaml
from pathlib import Path
import re


def validate_url_format(url: str) -> bool:
    """Validate URL format using basic regex."""
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url))


def validate_config(config_path: Path) -> int:
    """Validate external-sources.yml configuration.

    Returns:
        0 if valid, 1 if invalid
    """
    if not config_path.exists():
        print(f"Validation error: Config file not found: {config_path}")
        return 1

    try:
        content = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Validation error: Invalid YAML syntax: {e}")
        return 1
    except Exception as e:
        print(f"Validation error: Failed to read config file: {e}")
        return 1

    # Check if config is a dict
    if not isinstance(config, dict):
        print("Validation error: Config must be a dictionary")
        return 1

    # Check for 'sources' key
    if "sources" not in config:
        print("Validation error: Missing 'sources' key in config")
        return 1

    sources = config["sources"]

    # Check if sources is a list
    if not isinstance(sources, list):
        print("Validation error: 'sources' must be a list")
        return 1

    # Validate each source
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            print(f"Validation error: Source at index {i} must be a dictionary")
            return 1

        # Check required fields
        if "name" not in source:
            print(f"Validation error: Source at index {i} is missing 'name' field")
            return 1
        elif not source["name"]:
            print(f"Validation error: Source at index {i} has empty 'name' field")
            return 1

        if "url" not in source:
            print(f"Validation error: Source at index {i} is missing 'url' field")
            return 1
        elif not source["url"]:
            print(f"Validation error: Source at index {i} has empty 'url' field")
            return 1

        # Validate URL format
        url = source["url"]
        if not validate_url_format(url):
            print(
                f"Validation error: Source at index {i} has invalid URL format: {url}"
            )
            return 1

        # Check optional fields (branch and enabled)
        if "branch" in source:
            branch = source["branch"]
            if not isinstance(branch, str):
                print(
                    f"Validation error: Source at index {i} 'branch' must be a string"
                )
                return 1

        if "enabled" in source:
            enabled = source["enabled"]
            if not isinstance(enabled, bool):
                print(
                    f"Validation error: Source at index {i} 'enabled' must be a boolean"
                )
                return 1

    print("Config valid")
    return 0


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python3 validate_config.py <config_file>")
        print("Example: python3 validate_config.py .github/external-sources.yml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    sys.exit(validate_config(config_path))


if __name__ == "__main__":
    main()
