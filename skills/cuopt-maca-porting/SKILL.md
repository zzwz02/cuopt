---
name: cuopt-maca-porting
version: "26.06.00"
description: Port and debug NVIDIA cuOpt on MetaX C500 with MACA/cu-bridge. Use when working on cuOpt C++/CUDA build, link, runtime, CCCL compatibility, warp64, RAPIDS-free shims, CLI, or C API validation on MACA targets.
metadata:
  origin: skill-evolution
---

# cuOpt MACA Porting

Use this skill together with `cuopt-developer` when the task is about making cuOpt build or run on MetaX C500 through MACA/cu-bridge. Keep the scope anchored to the repository state in front of you; old porting notes can contain useful facts, but should not override current code or current build logs.

## Porting Baseline

- Treat the C500 MVP as C++ core, C API, and `cuopt_cli` first. Python, REST server, gRPC, conda, and wheel packaging are out of scope unless the user explicitly brings them back.
- Preserve the RAPIDS-free baseline under `cpp/vendor/include`. Do not reintroduce runtime dependencies on upstream `rmm`, `raft`, `rapids_logger`, `cudf`, or NVIDIA CCCL.
- Use MACA's bundled Thrust/CUB where possible. Prefer `thrust::cuda::par` for execution policy compatibility; add local shims only for missing `cuda::` / `cuda::std::` APIs that cuOpt actually uses.
- Remember that C500 warp size is 64. In warp semantics, prefer `raft::WarpSize`, `lane_mask_t`, and `LANE_MASK_ALL`; audit fixed `32`, `0xffffffff`, `uint32_t` ballot masks, and `& 31`. Keep genuine 32-thread non-warp kernels only after checking correctness.

## Build Model

- Source `maca_env.sh` before MACA commands. It should select the workspace CMake 3.30.x toolchain and set `MACA_PATH`, `CUCC_PATH`, and `CUCC_TARGETS`.
- Configure with `cmake_maca` and build with `ninja_maca` or `make_maca`. It is expected that configure-time CUDA compiler detection sees real NVIDIA `nvcc`; cu-bridge swaps to MACA tooling during the build.
- Keep the build directory at `cpp/build_maca` unless the user asks for a different one.
- For one-off CUDA probes, compile with `/opt/maca/tools/cu-bridge/tools/pre_make nvcc`. Do not call `cucc` directly for ad hoc single-file checks; direct `cucc` should stay inside `cmake_maca` / `ninja_maca` / `make_maca` flows.
- After linking, check `ldd` for MACA libraries such as `libmcblas`, `libmcsparse`, `libmcruntime`, `libmcsolver`, and `libmcrand`. CUDA imported-target full paths can bypass cu-bridge remapping; prefer explicit MACA library targets or remappable `-l` names when validating a MACA build.

## Runtime Gotchas

- MACA `mcsparseCreateDnVec` rejects null `values` even for zero-length dense vectors. When constructing cuSPARSE dense-vector descriptors from possibly empty device buffers, provide a non-null device dummy pointer and include a diagnostic label in descriptor creation errors.
- Treat `cuopt_cli` exit status and output together. Some CLI paths can print an error even when shell status looks benign; an output line like `Error in solve_lp` or `cuopt_cli error` is a failed smoke.
- MACA can warn that `__launch_bounds__(..., minBlocksPerMultiprocessor)` ignores the second argument. Treat it as a performance warning unless the failing test points at occupancy-sensitive correctness.
- Old MACA CCCL lacks some newer libcudacxx APIs. Prefer small local compatibility shims over fetching NVIDIA CCCL, and keep shims scoped to APIs proven missing by build errors.

## Validation Ladder

1. Run environment smoke: `source maca_env.sh`, `cmake_maca --version`, and `mx-smi`.
2. Build incrementally with `PARALLEL_LEVEL=1 ninja_maca -C cpp/build_maca cuopt_cli -j1`; use higher parallelism only after memory headroom is clear.
3. Confirm a no-op rebuild before runtime smoke: `ninja_maca -C cpp/build_maca cuopt_cli -j1` should print `ninja: no work to do`.
4. Run CLI LP smoke on `datasets/linear_programming/afiro_original.mps`; expect `Optimal` and objective near `-4.64753143e+02`.
5. Run QP/barrier smoke on `datasets/benchmarks/maros_meszaros/QAFIRO.QPS`; expect barrier `Optimal` and objective near `-1.59078161e+00`.
6. Build and run a filtered C API smoke before attempting all tests. The full `C_API_TEST` includes MIP and callback cases that can be long-running or outside the C500 MVP scope.
7. Record MIP, routing, server, or gRPC failures explicitly rather than silently skipping device functionality.
