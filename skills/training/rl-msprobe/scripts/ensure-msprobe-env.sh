#!/usr/bin/env bash
# Ensure msprobe (PyPI: mindstudio-probe) is importable in the current Python.
set -euo pipefail

TRY_INSTALL=1
for arg in "$@"; do
  case "$arg" in
    --no-install) TRY_INSTALL=0 ;;
    -h|--help)
      echo "Usage: $0 [--no-install]"
      echo "  Check msprobe import; by default run: pip install mindstudio-probe"
      exit 0
      ;;
  esac
done

check_import() {
  python3 -c "from msprobe.pytorch import PrecisionDebugger" >/dev/null 2>&1
}

print_ok() {
  python3 <<'PY'
try:
    import msprobe
    ver = getattr(msprobe, "__version__", None)
except Exception:
    ver = None
from msprobe.pytorch import PrecisionDebugger  # noqa: F401

if ver:
    print(f"msprobe OK (version={ver})")
else:
    print("msprobe OK")
PY
}

if check_import; then
  print_ok
  exit 0
fi

echo "msprobe is not available in the current Python environment." >&2
echo "  (import: from msprobe.pytorch import PrecisionDebugger)" >&2

if [[ "$TRY_INSTALL" -eq 1 ]]; then
  echo "Attempting: pip install mindstudio-probe" >&2
  if pip install mindstudio-probe; then
    if check_import; then
      print_ok
      exit 0
    fi
    echo "pip install succeeded but msprobe import still failed." >&2
  else
    echo "pip install mindstudio-probe failed." >&2
  fi
else
  echo "Skipped install (--no-install)." >&2
fi

cat >&2 <<'EOF'

Please install msprobe manually in the same environment used for verl training/inference, then verify:

  pip install mindstudio-probe
  python3 -c "from msprobe.pytorch import PrecisionDebugger; print('OK')"

If you use a container, run the above inside the container before dump collection.
EOF
exit 1
