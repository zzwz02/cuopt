#!/usr/bin/env bash
#
# Install MetaX's mcdf stack (cudf/rmm/cupy/numba for MACA/C500) into the active
# conda python3.10 env, so cuOpt's RAPIDS Python paths (routing + distance) run
# on C500. This is the routing/distance counterpart to
# build_lp_modules_rapids_free.sh (which builds the numpy-only LP/MILP modules).
#
# mcdf ships cp310 wheels with MIXED import names: mcdf->cudf, mcpy->cupy,
# but rmmx->rmmx and numbax->numbax. cuOpt's compiled routing .so do
# `from numba import cuda`, so we also install a numba->numbax site-packages
# shim. cupy's JIT must be pointed at MACA's cuda_fp16.h via CUDA_PATH.
#
# Usage:
#   source maca_env.sh
#   export PATH=/opt/conda/bin:$PATH          # conda python3.10 first
#   bash python/cuopt/setup_maca_rapids.sh /home/maca-mcdf-3.7.0.3-linux-x86_64.tar.xz
#
# Then run routing/distance with:
#   export CUDA_PATH=/opt/maca/tools/cu-bridge   # cupy online-JIT fp16 header
#   PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
#     python -m pytest python/cuopt/cuopt/tests/routing -q
set -euo pipefail

TARBALL="${1:-/home/maca-mcdf-3.7.0.3-linux-x86_64.tar.xz}"
[ -f "$TARBALL" ] || { echo "mcdf tarball not found: $TARBALL" >&2; exit 1; }

PY="$(command -v python3)"
echo "python3: $PY ($("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:2])))'))"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo "=== extracting cp310 wheels ==="
tar -xJf "$TARBALL" -C "$WORK" --wildcards 'mcdf-*/wheel/*cp310*.whl'
WHL="$(dirname "$(find "$WORK" -name 'mcdf-*cp310*.whl' | head -1)")"

echo "=== installing 4 MACA wheels (--no-deps) ==="
"$PY" -m pip install --no-deps --root-user-action=ignore \
  "$WHL"/rmmx-*cp310*.whl "$WHL"/mcpy-*cp310*.whl \
  "$WHL"/numbax-*cp310*.whl "$WHL"/mcdf-*cp310*.whl

echo "=== installing pure-python deps --no-deps skipped (cudf hard pins) ==="
"$PY" -m pip install --root-user-action=ignore \
  fastrlock 'llvmlite==0.39.1' 'pandas==1.5.3' 'pyarrow==10.0.1' 'protobuf==4.21.12'

echo "=== installing numba -> numbax site-packages shim ==="
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SP/numba"
cat > "$SP/numba/__init__.py" <<'PY'
"""MACA compatibility shim: redirect `import numba` to MetaX's numbax fork.

cuOpt's compiled routing/distance extensions do `from numba import cuda`, but on
MACA the numba fork is distributed as `numbax` (top-level package `numbax`, with
a working `numbax.cuda`). Alias numbax (and its submodules) under the `numba`
name in sys.modules so the unmodified cuOpt code resolves to MACA's numba.
"""
import sys
import importlib

_real = importlib.import_module("numbax")
importlib.import_module("numbax.cuda")

for _name, _mod in list(sys.modules.items()):
    if _name == "numbax" or _name.startswith("numbax."):
        sys.modules["numba" + _name[len("numbax"):]] = _mod

sys.modules["numba"] = _real
PY

echo "=== verifying stack on C500 ==="
CUDA_PATH=/opt/maca/tools/cu-bridge "$PY" - <<'EOF'
import cudf, rmmx
from numba import cuda
import cupy
assert int(cudf.Series([1, 2, 3]).sum()) == 6
assert cuda.is_available()
assert int((cupy.arange(5) + 1).sum().get()) == 15
print("OK: cudf", cudf.__version__, "| rmmx", rmmx.__version__,
      "| numba->numbax", cuda.is_available(), "| cupy JIT ok")
EOF
echo "ALL DONE. Remember to set CUDA_PATH=/opt/maca/tools/cu-bridge when running."
