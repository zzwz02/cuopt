# cuOpt Docker build — MetaX C500 & NVIDIA A100

One multi-stage `Dockerfile` produces two images from a **fresh clone of the cuOpt fork**
(`github.com/zzwz02/cuopt`, branch `feature/maca-port`):

| Image | Target platform | C++ build dir | CUDA path |
|---|---|---|---|
| `cuopt:c500` | MetaX C500 (MACA / cu-bridge, warp64) | `cpp/build_maca` (`-DCUOPT_MACA=ON`) | MACA cu-bridge |
| `cuopt:a100` | NVIDIA A100 | `cpp/build_cuda` | CUDA 12.9 |

## Design / constraints

- **Base image**: `sw-dockhub-zj.mxcr.io/xiangrong/ai-agent:20260506-690-torch2.4-py310-ubuntu22.04-amd64`
  (already ships CUDA 12.9, MACA + cu-bridge, conda python 3.10, ninja, git, g++).
- **No `COPY` from the working tree.** Source (cuOpt fork) and CCCL are `git clone`d, CMake
  3.30.8 is downloaded, all other toolchain comes from `apt`. Nothing local is injected.
- **Toolchain via apt** where possible: `ninja-build libboost-all-dev libtbb-dev libbz2-dev
  zlib1g-dev libsuitesparse-dev` (AMD ordering). Only CMake≥3.30 (binary) and CCCL 3.4.x
  (git, no public `v3.4.0` tag) are downloaded.
- **Datasets are not baked in** — mount the host home at run time (`-v ~/:/home`) so tests use
  `/home/cuopt-26.06-maca/datasets` and the mcdf tarball under `/home`.
- **Networking**: apt → in-image MetaX mirror (direct); pip → Aliyun mirror (direct); only
  github (clones + CMake FetchContent of papilo/pslp/gtest) uses the host proxy
  (`--build-arg PROXY=http://127.0.0.1:3578`, build with `--network=host`).

## Build

```bash
bash docker/build.sh          # both images (common -> c500 -> a100)
bash docker/build.sh c500     # only C500
bash docker/build.sh a100     # only A100
```

## Test (mirrors docs/dev/build_c500_a100.md §3–§6)

**C500** (exposes the MetaX device):

```bash
docker run --name cuopt-c500 -v ~/:/home \
  --device=/dev/mxcd --device=/dev/dri --gpus all \
  --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=5g --privileged --group-add video --network=host -it \
  cuopt:c500 bash /home/cuopt-26.06-maca/docker/test_c500.sh
```

**A100** (exposes the NVIDIA GPU; native RAPIDS / cudf 24.12 is baked into the image in an
isolated venv at build time, so nothing is installed at test time):

```bash
docker run --name cuopt-a100 -v ~/:/home --gpus all \
  --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=5g --network=host -it \
  cuopt:a100 bash /home/cuopt-26.06-maca/docker/test_a100.sh
```

Each test script runs: §3 the full C++ gtest matrix + examples, §4 the numpy-only
LP/MILP/QP pytest, §5 routing/distance pytest (C500: mcdf; A100: native RAPIDS venv),
§6 a REST server smoke (health + async routing solve retrieved via msgpack).
