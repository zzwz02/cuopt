#!/usr/bin/env bash
#
# Build the cuOpt C500 and A100 images from the Dockerfile in this directory.
# Nothing from the working tree is sent into the build except the Dockerfile;
# all sources (cuOpt fork, CCCL) and toolchain are fetched during the build.
#
# Usage:
#   bash docker/build.sh            # build both (common -> c500 -> a100)
#   bash docker/build.sh c500       # build only the C500 image
#   bash docker/build.sh a100       # build only the A100 image
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY="${PROXY:-http://127.0.0.1:3578}"          # host clash proxy (github access)
CUOPT_REF="${CUOPT_REF:-feature/maca-port}"
CUOPT_REPO="${CUOPT_REPO:-https://github.com/zzwz02/cuopt.git}"

# Minimal, empty build context (we COPY nothing).
CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT

build() {
  local target="$1" tag="$2"
  echo "=================================================================="
  echo "  docker build  target=$target  ->  $tag"
  echo "=================================================================="
  docker build \
    --network=host \
    --target "$target" \
    --build-arg PROXY="$PROXY" \
    --build-arg CUOPT_REPO="$CUOPT_REPO" \
    --build-arg CUOPT_REF="$CUOPT_REF" \
    -f "$HERE/Dockerfile" \
    -t "$tag" \
    "$CTX"
}

what="${1:-all}"
case "$what" in
  c500) build c500 cuopt:c500 ;;
  a100) build a100 cuopt:a100 ;;
  all)  build c500 cuopt:c500 ; build a100 cuopt:a100 ;;
  *) echo "usage: $0 [c500|a100|all]" >&2; exit 2 ;;
esac
echo "DONE: $what"
