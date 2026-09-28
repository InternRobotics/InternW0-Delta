#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_ROOT="${WAM_LIBERO_REPO_ROOT:-${REPO_ROOT}/third_party/LIBERO-plus}"
PYTHON="${PYTHON:-python}"
REVISION=4976dc30028e805ff8094b55501d532c48fec182
if [[ ! -e "$LIBERO_ROOT" ]]; then
  mkdir -p "$(dirname "$LIBERO_ROOT")"
  git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_ROOT"
  git -C "$LIBERO_ROOT" checkout --detach "$REVISION"
fi
[[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" == "$REVISION" ]] || {
  echo "Expected LIBERO Plus revision $REVISION at $LIBERO_ROOT" >&2; exit 1;
}
# Initial states contain Python/numpy objects from the trusted benchmark.
"$PYTHON" - "$LIBERO_ROOT" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]) / "libero/libero/benchmark/__init__.py"
s = p.read_text().replace("torch.load(init_states_path)", "torch.load(init_states_path, weights_only=False)")
p.write_text(s)
PY
"$PYTHON" -m pip install --no-deps -e "$LIBERO_ROOT"
echo "LIBERO Plus installed at $LIBERO_ROOT. Download its assets as described in eval/libero_plus/README.md."
