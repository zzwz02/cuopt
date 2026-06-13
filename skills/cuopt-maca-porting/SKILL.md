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
- Always run C500 builds/tests with `export MACA_DIRECT_DISPATCH=1` (now set by `maca_env.sh`). MACA's default deferred/queued kernel dispatch causes cross-kernel ordering hangs and wrong results on C500 — e.g. PDLP batch tests (`simple_batch_*`) fail or the suite hangs when many solves run back-to-back in one process. Direct dispatch makes the launch path deterministic and is required for cuOpt correctness, not just performance.
- Configure with `cmake_maca` and build with `ninja_maca` or `make_maca`. It is expected that configure-time CUDA compiler detection sees real NVIDIA `nvcc`; cu-bridge swaps to MACA tooling during the build.
- Keep the build directory at `cpp/build_maca` unless the user asks for a different one.
- For one-off CUDA probes, compile with `/opt/maca/tools/cu-bridge/tools/pre_make nvcc`. Do not call `cucc` directly for ad hoc single-file checks; direct `cucc` should stay inside `cmake_maca` / `ninja_maca` / `make_maca` flows.
- After linking, check `ldd` for MACA libraries such as `libmcblas`, `libmcsparse`, `libmcruntime`, `libmcsolver`, and `libmcrand`. CUDA imported-target full paths can bypass cu-bridge remapping; prefer explicit MACA library targets or remappable `-l` names when validating a MACA build.

## Runtime Gotchas

- MACA `mcsparseCreateDnVec` rejects null `values` even for zero-length dense vectors. When constructing cuSPARSE dense-vector descriptors from possibly empty device buffers, provide a non-null device dummy pointer and include a diagnostic label in descriptor creation errors.
- Treat `cuopt_cli` exit status and output together. Some CLI paths can print an error even when shell status looks benign; an output line like `Error in solve_lp` or `cuopt_cli error` is a failed smoke.
- MACA can warn that `__launch_bounds__(..., minBlocksPerMultiprocessor)` ignores the second argument. Treat it as a performance warning unless the failing test points at occupancy-sensitive correctness.
- Old MACA CCCL lacks some newer libcudacxx APIs. Prefer small local compatibility shims over fetching NVIDIA CCCL, and keep shims scoped to APIs proven missing by build errors.
- Watch for **read-before-write device buffers** that are only safe because of fresh/zeroed allocations. On C500 the rmm pool readily hands back memory freed by a previous solve, so a buffer that is read before being fully written reads *that solve's garbage* instead of zeros — a cross-test/cross-solve contamination that passes in isolation but fails when a prior solve ran in the same process. Concretely: PDLP's saddle-point `current_AtY_`/`next_AtY_` (`cpp/src/pdlp/saddle_point.cu`) were left un-zeroed ("written by SpMV"), but in **batch** mode `update_solution` swaps them each step so the reflected-primal projection can read entries the current SpMM didn't rewrite; after a per-climber-objective batch solve freed its buffers, `simple_batch_different_constraint_bounds` / `simple_batch_everything_different` diverged to the iteration limit. Fix = zero-init both AtY buffers in the constructor. When a batch/PDLP test fails or hangs **only after another solve ran first**, suspect an uninitialized buffer, not the test. (`MACA_DIRECT_DISPATCH=1` does not mask these.)
- Cooperative grid launches (`cudaLaunchCooperativeKernel` + `cg::this_grid().sync()`) are unreliable on C500 and can hang the GPU at 100% util with no progress. Root cause: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` only reports the warp/thread-limited occupancy (e.g. 16 for a 128-thread block at 2048 threads/SM) and ignores register/shared-mem pressure, **and** the true co-residency for a cooperative launch is even lower than that (empirically ~13-15/SM for a register-heavy kernel where the API says 16). The upstream idiom `grid = numSM * cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)` therefore over-sizes the grid; the surplus blocks never become resident, and every `cg::this_grid().sync()` (and any hand-rolled global-memory arrival-counter barrier — same mechanism) deadlocks waiting on them. Unlike NVIDIA, MACA's `cudaLaunchCooperativeKernel` does **not** validate the grid size (NVIDIA returns `cudaErrorCooperativeLaunchTooLarge`), so the oversize silently hangs instead of erroring. This bit PDLP's PDLP_TEST `run_sub_mittleman` on large instances (e.g. `graph40-40`), but only in the `Methodical1` solver mode, which is the sole mode using `restart_strategy = 2` (`TRUST_REGION_RESTART`) and its `solve_bound_constrained_trust_region_kernel`. Small problems clamp the grid to 1 block and survive; large problems use the full bogus grid and hang. There is no reliable way to query the safe cooperative grid size on C500, so do not depend on it: prefer **host-driven multi-kernel** loops where each grid-sync point becomes a kernel-launch boundary (regular, non-cooperative launches at any grid size — blocks run in waves, kernel completion is the global sync). A single-block cooperative launch also avoids the deadlock but is far too slow (MACA's `cg::this_grid().sync()` has high per-call latency: graph40-40 went 0.3s → 7.2s, scpm1 → 69s).

## Validation Ladder

1. Run environment smoke: `source maca_env.sh`, `cmake_maca --version`, and `mx-smi`.
2. Build incrementally with `PARALLEL_LEVEL=1 ninja_maca -C cpp/build_maca cuopt_cli -j1`; use higher parallelism only after memory headroom is clear.
3. Confirm a no-op rebuild before runtime smoke: `ninja_maca -C cpp/build_maca cuopt_cli -j1` should print `ninja: no work to do`.
4. Run CLI LP smoke on `datasets/linear_programming/afiro_original.mps`; expect `Optimal` and objective near `-4.64753143e+02`.
5. Run QP/barrier smoke on `datasets/benchmarks/maros_meszaros/QAFIRO.QPS`; expect barrier `Optimal` and objective near `-1.59078161e+00`.
6. Build and run a filtered C API smoke before attempting all tests. The full `C_API_TEST` includes MIP and callback cases that can be long-running or outside the C500 MVP scope.
7. Record MIP, routing, server, or gRPC failures explicitly rather than silently skipping device functionality.
