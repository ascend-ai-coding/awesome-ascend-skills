"""Capture a test command without a shell and propagate its exit status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('provide a command after --')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'command': command, 'cwd': str(Path.cwd()),
              'runner_python': sys.executable, 'child_exit_code': None}
    try:
        with (output / 'stdout.txt').open('wb') as stdout, \
                (output / 'stderr.txt').open('wb') as stderr:
            child = subprocess.run(command, stdout=stdout, stderr=stderr,
                                   shell=False)
        result['child_exit_code'] = child.returncode
        status = child.returncode if child.returncode >= 0 else 128 - child.returncode
    except OSError as error:
        result['launch_error'] = str(error)
        status = 127
    result['runner_exit_code'] = status
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
