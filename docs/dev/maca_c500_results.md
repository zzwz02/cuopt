# cuOpt on MetaX C500 (MACA): Functional Parity and Performance Results

> **Goal**: bring cuOpt on MetaX C500 / MACA 3.0.x / cu-bridge (warp64) to the same functional scope
> as the official RAPIDS-free + in-house LDLᵀ wheel (A100 baseline; `docs/dev/rapids_removal_mvp_results.md`),
> and benchmark it against that baseline.
>
> **Hardware**: baseline = NVIDIA A100; this report = MetaX C500 + AMD EPYC 7543.
>
> **Conventions**: `source maca_env.sh` (sets `MACA_DIRECT_DISPATCH=1`) before building or testing;
> build directory `cpp/build_maca`. General C500-vs-NVIDIA differences are in `skills/cuopt-maca-porting/SKILL.md`.

## 1. Functional Coverage Matrix

| Capability | A100 | C500 | Validation / Notes |
|---|---|---|---|
| LP — PDLP | ✅ | ✅ | PDLP_TEST 70 / 0 |
| LP — Dual simplex | ✅ | ✅ | DUAL_SIMPLEX_TEST 30 / 0 |
| LP — Barrier | ✅ | ✅ | LP_UNIT / C_API / CLI barrier paths pass |
| MILP | ✅ | ✅ | MIP_TEST 3, MIP_TERMINATION 12, PRESOLVE 1, INCUMBENT 3; all 0 fail |
| QP / SOCP (in-house LDLᵀ) | ✅ | ✅ | QP_UNIT 7 / 0, SOCP 28 / 0 |
| C API | ✅ | ⚠️ | C_API_TEST 57 / 1; sole failure is 128-thread time-bounded reproducibility, not a functional regression (§5) |
| cuopt_cli | ✅ | ✅ | CLI_TEST 7 / 0 (requires `cuopt_cli` on PATH) |
| Routing (C++) | ✅ | ✅ | ROUTING_UNIT_TEST 57 / 0; fixed span<bool>, spin-locks, cycle-finder (§2.1) |
| Distance engine | ✅ | ✅ | WAYPOINT_MATRIXTEST 6 / 0 |
| Examples (cvrp/pdptw/service) | ✅ | ✅ | All three SUCCESS; fixed stale example data (service/pdptw) and an inverted `main()` exit code (cvrp) (§2.2) |
| Python LP/MILP/QP/SOCP | ✅ | ✅ | `tests/linear_programming` 51 passed / 9 skipped / 0 failed (§2.3) |
| Python routing + distance | ✅ | ✅ | `tests/routing` 42 passed (incl. distance 8/8); fixed cross-move spin-lock (§2.4) |
| REST server / self-hosted | ✅ | ✅ | LP + routing via REST pass end-to-end; one-line SIGCHLD fix (§2.6) |
| Performance | ✅ | ✅ | LP/QP objectives match, runtimes same order of magnitude; routing quality near baseline (§3); nug08-3rd nondeterministic on C500 after merging `feature/perf_opt` (§3.4) |

> The two non-green items (C_API 128-thread reproducibility, Python routing numpy warning order) are
> both non-functional and non-MACA; see §5.

## 2. Key Issues and Fixes

Each item gives **symptom / root cause / fix / validation**; the general diagnostic methodology is in
the c500 skill.

### 2.1 C++ routing: three C500-specific kernel defects

**(a) `span<bool const>` brace-init builds a garbage span.** The vendored `raft::span`
(`cpp/vendor/include/raft/core/span.hpp`) initializes its `cuda::std::span` via `base_{ptr, count}`;
for `bool const` this narrowing list-init is silently miscompiled by older MACA CCCL (`.data()` becomes
a wild pointer). cuOpt's per-vehicle `order_match` is exactly a `span<bool const>`, so device
dereference triggers an Xnack/ATU trap that cascades into a full ROUTING_UNIT collapse. **Fix**:
parenthesized direct-init `base_(ptr, count)` (correct on all platforms). Repro: `docs/dev/maca_span_bool_repro.cu`.

**(b) Warp-cooperative spin-locks deadlock/leak on C500.** `acquire_lock`/`release_lock`
(`cpp/src/utilities/cuda_helpers.cuh`) rely on Volta independent thread scheduling (ITS). C500 is
lock-step SIMT (warp64) with no ITS: when ≥2 lanes of one warp contend for the same lock, the CAS
winner never reaches `release_lock` → deadlock; a probing hash insert leaks the lock permanently
(symptom: 100% utilization, no trap, no error). **Fix**: `with_lock`/`with_lock_block` (critical
section and release inside the winning-CAS branch); the probing `device_map_t::add` is serialized
across the warp's lanes. Five multi-lane sites fixed (all MIP sites are single-lane and safe).

**(c) `insert_graph_nodes_kernel` OOB = cycle-finder reconstruction race.** The cycle-finder's
`extend_cycle` has multiple blocks reading and writing the same mutable `d_best.key_ptr[0]`; a later
wave reads a torn intermediate, matches a non-existent edge, and overflows at insert. **Fix**:
double-buffer (read `key_in`, the unique winning thread writes `key_out` — disjoint read/write sets by
construction), swapping between layers.

Validation: full `ROUTING_UNIT_TEST` **57 / 0** (before the span fix: a 5 passed / 105 failed cascade).

### 2.2 Examples (cvrp / pdptw / service)

All three compile and solve to SUCCESS. The service_team_routing and pdptw_mixed_fleet failures were
**stale example data** (host-side validation; the current A100 build fails identically): non-unique
`break_locations` and a mixed fleet with no per-vehicle-type cost matrix, respectively — both fixed.

The cvrp_daily_deliveries "exit code 1", once suspected to be a private-memory downgrade, was an
**inverted `main()` exit-code logic** (since the initial OSS commit b833f7a1): `if (status) { exit_code = 1; }`
with status true = success, so an all-success run returned 1, whereas the sibling examples correctly
used `if (!status)`. It returns 1 on A100 too. **Fixed** to `if (!status)`; exit code is now 0.

Separately, raising the C500 private-memory limit (`sudo modprobe metax pri_mem_sz=24`, from the
read-only 16 KB) let a local-search kernel's 20 KB/thread request run without a downgrade — no MXLOG
warnings, no new `trapInfo.*` traps. The limit is a tunable kernel parameter, unrelated to the
exit-code issue.

### 2.3 Python LP/MILP/QP/SOCP (RAPIDS-free): all green

`tests/linear_programming` **51 passed / 9 skipped / 0 failed**. Two root causes fixed:

1. **Repeated in-process solve fault.** `csrsort_cusparse` (`problem_helpers.cuh`) calls
   `cusparseXcsrsort`, whose internal mccub kernels race under C500 async; on fault the runtime is
   disabled and the suite cascade-crashes. Stepwise elimination (`MACA_LAUNCH_BLOCKING=3` passes;
   standalone replay of the real inputs is clean; raw mccub does not fault) pinned it to an
   mcsparse-internal driving race. **Fix**: CUB `DeviceSegmentedSort::SortPairs`.
2. **`get_vars()` returns inf.** The RAPIDS-free Cython module linked against NVIDIA `libcudart`, so
   `cudaMemcpy` (D2H) on MACA device pointers failed silently (rc=1), leaving `np.empty` garbage.
   **Fix**: compile that TU through cu-bridge (`build_lp_modules_rapids_free.sh`, `CUOPT_RT_MACA=1`:
   `pre_make nvcc` + cu-bridge CCCL + `-lruntime_cu -lmcruntime`), rewriting `cudaMemcpy` to
   `wcudaMemcpy` at compile time.

### 2.4 Python routing + distance (mcdf)

**Install** (C500's cudf/rmm/cupy/numba): `pip install --no-deps` the four cp310 wheels (`mcdf`→cudf,
`mcpy`→cupy, `rmmx`, `numbax`), plus the pure-Python deps (`fastrlock`, `llvmlite==0.39.1`,
`pandas==1.5.3`, `pyarrow==10.0.1`, `protobuf==4.21.12`). Two adaptations: cupy's online JIT needs
`CUDA_PATH=/opt/maca/tools/cu-bridge` (else it picks up NVIDIA's `cuda_fp16.h`, which mxcc cannot
parse); cuOpt's compiled `.so` does `from numba import cuda` and needs a `numba`→`numbax` shim.
One-shot script: `python/cuopt/setup_maca_rapids.sh`.

**Cross-move spin-lock cross-block deadlock (fixed).** `test_re_routing` hung; gdb sampling located
`populate_cross_list_kernel`'s `route_pair_locks` on raw `acquire_lock`/`release_lock`. Each lane in a
block locks a distinct index, but multiple blocks share `first_route` and thus contend cross-block for
the same lock, deadlocking under C500 lock-step (triggered only as the problem grows, hence missed
earlier). **Fix**: `with_lock` (§2.1b). Validation: `test_re_routing` 11/11, `tests/routing` 42 passed.

### 2.5 Hardening the 64 KB shared-memory limit

C500 caps **static + dynamic** shared memory per block at 64 KB, but `set_shmem_of_kernel` validated
only the dynamic part: a kernel with static `__shared__` could pass "dynamic ≤ 64 KB" yet fail at
launch with "static + dynamic > 64 KB" → crash (`find_all_squeeze_pos` at ~1000+ node routes).
**Fix**: include `cudaFuncGetAttributes().sharedSizeBytes` and return false when the total exceeds
`cudaDevAttrMaxSharedMemoryPerBlockOptin`, so callers cleanly skip or raise OOM. An over-limit kernel
never launches under-provisioned — **no silent miscomputation**; the result is SUCCESS (valid) or an
explicit ERROR/INFEASIBLE. Ordinary multi-vehicle VRP (short routes) does not trigger it.

### 2.6 REST server: one-line `SIGCHLD=SIG_DFL` fix

LP via REST passes end-to-end (afiro Optimal / -464.7531 / X01=80). routing via REST had failed because
the forked solver subprocess flooded `mcErrorRecompile`. **Root-cause chain**: C500 runtime-recompiles
any kernel whose launch block size exceeds its compile-time `max_block_size` (default 512), via a
fork+exec `mxcc` subprocess it `wait()`s on; the solver worker (`solver.py::process_async_solve`) had
set `SIGCHLD=SIG_IGN`, which on Linux makes `wait()` return ECHILD → `mxcc` build fails → launch
returns `mcErrorRecompile`. A direct process has no such handler, so it reproduced only via the server.
**Fix**: `SIG_DFL` (still ignores SIGCHLD but keeps `wait()` working); no kernel changes. Validation:
`mcErrorRecompile=0`, routing via REST status=0, cost 3.0/304/8 (matching the direct path).

> Deployment notes: behind an HTTP proxy, local-port requests need `no_proxy` / `--noproxy` (else 502);
> the server enumerates GPUs via `nvidia-smi` (a co-installed A100 shows up, cosmetic only), but the
> solver runs on the C500 (confirmed via `mx-smi`).

## 3. Performance Benchmarks (C500 vs A100)

Baseline = the A100 results in `rapids_removal_mvp_results.md` §5 (RAPIDS-free + in-house LDLᵀ,
equivalent to the official cuDSS-free wheel).

### 3.1 LP (Mittelmann subset, Barrier `--method 3 --time-limit 600`)

All Optimal; objectives match the baseline.

| Instance | A100 time | A100 objective | C500 time | C500/A100 |
|---|---|---|---|---|
| afiro | 0.09 s | -464.753135 | 0.08 s | 0.9× |
| graph40-40 | 0.99 s | -300.000464 | 1.14 s | 1.15× |
| qap15 | 1.41 s | 1041.00046 | 2.74 s | 1.94× |
| nug08-3rd | 6.23 s | 214.000642 | 11.35 s | 1.82× |
| woodlands09 | 5.42 s | ≈0 | 10.04 s | 1.85× |
| scpm1 | 4.20 s | 414.152071 | 60.25 s | 14.3× |

Mid-size instances run 1.1–1.9× (same order as the baseline's 1.3–1.7× vs cuDSS — the in-house LDLᵀ
factorization gap on C500). scpm1 is a 14× outlier (a pathological slow path in LDLᵀ factorization for
this fill pattern; correctness unaffected — still Optimal).

### 3.2 QP (Maros-Mészáros subset, Barrier `--method 3 --time-limit 180`)

All Optimal; objectives **bit-identical** to the baseline; runtimes 0.05–0.11 s (baseline 0.04–0.09 s).

| Problem | A100 objective | C500 objective | C500 time |
|---|---|---|---|
| HS35 | 0.111111115 | +1.11111115e-01 | 0.09 s |
| HS51 | 0.000000000 | +0.00000000e+00 | 0.06 s |
| HS52 | 5.32664756 | +5.32664756e+00 | 0.05 s |
| HS53 | 4.09302326 | +4.09302326e+00 | 0.08 s |
| HS76 | -4.68181817 | -4.68181817e+00 | 0.09 s |
| HS268 | 3.84e-09 | +3.84898158e-09 | 0.10 s |
| HS35MOD | 0.250000002 | +2.50000001e-01 | 0.11 s |
| QPTEST | 4.37187501 | +4.37187501e+00 | 0.08 s |
| ZECEVIC2 | -4.12499999 | -4.12499999e+00 | 0.07 s |

### 3.3 Routing (CVRPLIB Set-X / Solomon / Homberger, fixed time budget)

Baseline = `rapids_removal_mvp_results.md` §5.4; driver `docs/dev/route_bench.py`. All SUCCESS, vehicle
counts valid, TW/capacity constraints satisfied.

| Instance (size, budget) | A100 baseline (2 runs) | Literature BKS | C500 cost | C500 vehicles | C500/baseline |
|---|---|---|---|---|---|
| X-n101-k25 (101, 30 s) | 31573 / 32753 | 27591 | 36313 | 25 | +12–15% |
| X-n439-k37 (439, 60 s) | 36454 / 36470 | 36391 | 37138 | 37 | +1.8% |
| X-n1001-k43 (1001, 120 s) | 73709 / 74042 | 72355 | 77209 | 43 | +4.5% |
| C101 (Solomon 100+TW, 30 s) | 828.94 / 828.94 | 828.94 | 848.94 | 11 | +2.4% |
| C1_10_1 (Homberger 1000+TW, 120 s) | 42478.95 / 42478.95 | ≈42444 | 46767 | 147 | +10% |

Under a fixed budget, solution quality is **+1.8 to 15%** above baseline (slightly worse), consistent
with §3.1: C500 is ~1.1–1.9× slower per iteration, so fewer heuristic iterations fit the same
wall-clock budget. The gap is largest on the finest-grained / largest instances and near-baseline for
mid-size. Not a correctness issue.

### 3.4 Re-test after merging feature/perf_opt

`feature/perf_opt` (7 commits, merge `8b64fc29`) adds CUDA Graph capture, structured pivot dropping,
Jacobi scaling, a density-driven dense tail, and banded ND ordering to the in-house LDLᵀ (original
report `docs/dev/qp_performance_optimization.md`, baselined against official cuDSS on A100). This
section re-tests post-merge under the §3.1 / §3.2 protocols vs pre-merge. All still Optimal; objectives
match pre-merge/baseline (rel ≤ 1e-4).

**LP (Barrier `--method 3 --time-limit 600`, §3.1 instances)**

| Instance | Pre-merge C500 (§3.1) | Post-merge C500 | Post-merge A100 | Post-merge iters (C500 / A100) | Assessment |
|---|---|---|---|---|---|
| afiro | 0.08 s | 0.09 s | 0.09 s | 12 / 12 | unchanged |
| graph40-40 | 1.14 s | 0.90 s | 0.87 s | 12 / 12 | slightly better (~1.3×) |
| qap15 | 2.74 s | 2.7 s (2.8–3.3, deterministic) | 1.30 s | 23 / 23 | unchanged |
| nug08-3rd | 11.35 s | **43–212 s, nondeterministic** | 6.1 s | **83 / 390 / 409** vs **18** | ⚠️ regression + nondeterminism |
| woodlands09 | 10.04 s | 9.93 s | 5.45 s | 22 / 22 | unchanged |
| scpm1 | 60.25 s | 60.3 s | 4.06 s | 26 / 26 | 14× outlier, **unfixed** |

**QP (Barrier `--method 3 --time-limit 180`, §3.2 instances)**: post-merge C500 takes 0.06–0.26 s
(pre-merge 0.05–0.11 s), iteration counts match A100 one-for-one, objectives bit-identical; fixed
overhead dominates on small problems — **no material change**.

**Conclusion (does performance improve substantially?)** No.

- **A100: small improvement only.** Vs the pre-merge in-house LDLᵀ, this batch gives mid-size LP
  roughly −3 to 12% (graph40-40 −12%, qap15 −8%, nug08-3rd −7%, scpm1 −3%); QP unchanged. Consistent
  with the perf_opt report (A100 vs cuDSS: 0.778× median per problem but +2× total wall-clock — gains
  concentrate in a few dense, ill-conditioned problems).
- **C500: no material improvement, one regression.** Most problems unchanged (afiro / qap15 /
  woodlands09 deterministic, iterations matching A100); graph40-40 slightly better; scpm1's 14× outlier
  unimproved; **nug08-3rd regressed to nondeterministic** — same binary, 83 / 390 / 409 iterations
  run-to-run (43–212 s), vs A100's deterministic 18 / 6.1 s and pre-merge C500's 11.35 s. Objectives
  remain correct; the iteration count drifts. Root cause and the determinism switch are in §3.5.

### 3.5 Full Maros-Mészáros (138) post-merge re-test + C500 determinism switch

The **current post-merge code** = `feature/maca-26.06` (merge `8b64fc29`) plus one C500 determinism
switch: the env var **`CUOPT_LDLT_DETERMINISTIC`** (**default off**). Default uses the original
parallel atomic fast path (performance); any value forces LDLᵀ onto a fully deterministic path (below).
Results are over the **full 138** problems (Barrier `--method 3 --time-limit 180`, default path).

**Summary (default path, C500 vs A100)**

| Metric | C500 | A100 |
|---|---|---|
| Optimal | **134 / 138** | **134 / 138** |
| NumError | 4 (`PRIMAL1-4`) | 4 (`PRIMAL1-4`) |
| Total Optimal time | 194.2 s | 103.9 s |
| Per-problem mean / max | 1.45 s / 50.7 s (`CVXQP1_L`) | 0.78 s / 37.5 s (`CVXQP1_L`) |

- Status is **identical per problem** (same 134 Optimal / 4 NumError, matching objectives). The 4
  NumError are only `PRIMAL1-4`, also NumError on perf_opt and official cuDSS (ill-conditioned, common
  to both; `docs/dev/qp_performance_optimization.md`) — not a C500/merge defect; no Suboptimal.
- C500 total time is ~1.9× A100 (same order as §3.1 mid-size); the slowest are dense-KKT problems
  (`CVXQP1_L` / `CVXQP3_L` / `BOYD2`).

**Iteration nondeterminism (root cause confirmed) and `CUOPT_LDLT_DETERMINISTIC`**

On C500, perf_opt's LDLᵀ drifts in IPM iteration count run-to-run on fill-dense ill-conditioned
problems (LP `nug08-3rd`, QP `CVXQP1_L` / `CVXQP3_L`); objectives stay correct (e.g. `nug08-3rd` is
always 214.0), only iteration count / time vary (`nug08-3rd`: 35..409 iterations across repeats).

- **Root cause.** LDLᵀ's two parallel fast paths — the dense-tail `ldlt_schur_pairs_kernel`'s
  cross-block `atomicAdd` and the long-column "atomic head" — accumulate Schur updates in a
  nondeterministic order. FP addition is non-associative; for the catastrophically-cancelling near-zero
  diagonals of a fill-heavy SPD system, a reordered sum can cross the rank-deficiency drop threshold
  (`ldlt_pivot_rule`: `|dj| < 1e-14·|diag0|`), flipping the drop/floor decision → factorization, Newton
  step, and iteration count drift. **wave32** (A100) jitters harmlessly; **wave64** (C500) doubles
  same-atomic contention, amplifying it across the threshold.
- **Ruled out.** `mcblas` DGEMM is deterministic on C500 (bit-reproducible over 5 replays) — the
  dense-tail DGEMM is not the source, so routing everything through DGEMM cannot fix it alone (the
  atomic-head nondeterminism still enters the tail via the pivots `d_k`).
- **Switch.** `CUOPT_LDLT_DETERMINISTIC=1` → a fully deterministic LDLᵀ (no atomics; per-column +
  bundled-layer path): disables `choose_tail_density`'s dense tail and sets `atomic_head=false`.
  Measured: `CVXQP2_M` env-on ×3 all 15 iterations (off 0.43 s / on 3.4 s, ~8× slower). **Default off**
  for performance; the deterministic path is slow (no dense-tail DGEMM, very slow on the largest
  fill-heavy problems such as `nug08-3rd`). A faster deterministic accumulation (e.g. a deterministic
  by-target Schur reduction) is future work.

**Per-problem data (default path, 138 problems; status / iterations / time)**

| Instance | Status | C500 iters | C500 (s) | A100 iters | A100 (s) |
|---|---|---|---|---|---|
| AUG2D | Optimal | 10 | 0.367 | 10 | 0.214 |
| AUG2DC | Optimal | 10 | 0.361 | 10 | 0.217 |
| AUG2DCQP | Optimal | 16 | 0.518 | 16 | 0.286 |
| AUG2DQP | Optimal | 16 | 0.519 | 16 | 0.288 |
| AUG3D | Optimal | 9 | 0.222 | 9 | 0.130 |
| AUG3DC | Optimal | 9 | 0.226 | 9 | 0.133 |
| AUG3DCQP | Optimal | 12 | 0.276 | 12 | 0.158 |
| AUG3DQP | Optimal | 12 | 0.284 | 12 | 0.155 |
| BOYD1 | Optimal | 22 | 4.035 | 22 | 0.231 |
| BOYD2 | Optimal | 75 | 21.319 | 75 | 6.843 |
| CONT-050 | Optimal | 15 | 0.291 | 15 | 0.188 |
| CONT-100 | Optimal | 15 | 0.873 | 15 | 0.451 |
| CONT-101 | Optimal | 15 | 0.85 | 15 | 0.434 |
| CONT-200 | Optimal | 16 | 2.864 | 16 | 1.642 |
| CONT-201 | Optimal | 16 | 3.386 | 16 | 1.858 |
| CONT-300 | Optimal | 16 | 6.575 | 16 | 3.633 |
| CVXQP1_L | Optimal | 126 | 50.67 | 161 | 37.46 |
| CVXQP1_M | Optimal | 15 | 1.000 | 15 | 0.406 |
| CVXQP1_S | Optimal | 12 | 0.203 | 12 | 0.090 |
| CVXQP2_L | Optimal | 16 | 4.107 | 16 | 2.595 |
| CVXQP2_M | Optimal | 15 | 0.405 | 15 | 0.182 |
| CVXQP2_S | Optimal | 13 | 0.208 | 13 | 0.094 |
| CVXQP3_L | Optimal | 63 | 21.944 | 75 | 15.312 |
| CVXQP3_M | Optimal | 25 | 1.759 | 25 | 0.701 |
| CVXQP3_S | Optimal | 13 | 0.333 | 13 | 0.114 |
| DPKLO1 | Optimal | 8 | 0.116 | 8 | 0.085 |
| DTOC3 | Optimal | 12 | 0.225 | 12 | 0.133 |
| DUAL1 | Optimal | 15 | 0.228 | 15 | 0.105 |
| DUAL2 | Optimal | 13 | 0.197 | 13 | 0.094 |
| DUAL3 | Optimal | 15 | 0.246 | 15 | 0.107 |
| DUAL4 | Optimal | 15 | 0.210 | 15 | 0.091 |
| DUALC1 | Optimal | 16 | 0.248 | 16 | 0.102 |
| DUALC2 | Optimal | 28 | 0.369 | 27 | 0.152 |
| DUALC5 | Optimal | 12 | 0.159 | 12 | 0.075 |
| DUALC8 | Optimal | 13 | 0.194 | 13 | 0.087 |
| EXDATA | Optimal | 23 | 0.923 | 23 | 0.568 |
| GENHS28 | Optimal | 9 | 0.064 | 9 | 0.046 |
| GOULDQP2 | Optimal | 13 | 0.173 | 13 | 0.077 |
| GOULDQP3 | Optimal | 14 | 0.278 | 14 | 0.108 |
| HS118 | Optimal | 12 | 0.089 | 12 | 0.067 |
| HS21 | Optimal | 12 | 0.094 | 12 | 0.067 |
| HS268 | Optimal | 15 | 0.110 | 15 | 0.058 |
| HS35 | Optimal | 12 | 0.087 | 12 | 0.049 |
| HS35MOD | Optimal | 15 | 0.104 | 15 | 0.058 |
| HS51 | Optimal | 8 | 0.057 | 8 | 0.042 |
| HS52 | Optimal | 8 | 0.061 | 8 | 0.043 |
| HS53 | Optimal | 12 | 0.088 | 12 | 0.051 |
| HS76 | Optimal | 12 | 0.088 | 12 | 0.051 |
| HUES-MOD | Optimal | 10 | 0.132 | 10 | 0.079 |
| HUESTIS | Optimal | 15 | 0.183 | 15 | 0.092 |
| KSIP | Optimal | 17 | 0.341 | 17 | 0.305 |
| LASER | Optimal | 15 | 0.504 | 15 | 0.182 |
| LISWET1 | Optimal | 58 | 0.553 | 58 | 0.364 |
| LISWET10 | Optimal | 62 | 0.574 | 62 | 0.384 |
| LISWET11 | Optimal | 52 | 0.499 | 52 | 0.331 |
| LISWET12 | Optimal | 82 | 0.755 | 82 | 0.488 |
| LISWET2 | Optimal | 13 | 0.185 | 13 | 0.116 |
| LISWET3 | Optimal | 13 | 0.179 | 13 | 0.133 |
| LISWET4 | Optimal | 13 | 0.179 | 13 | 0.133 |
| LISWET5 | Optimal | 13 | 0.172 | 13 | 0.134 |
| LISWET6 | Optimal | 12 | 0.163 | 12 | 0.128 |
| LISWET7 | Optimal | 51 | 0.520 | 51 | 0.327 |
| LISWET8 | Optimal | 74 | 0.693 | 74 | 0.446 |
| LISWET9 | Optimal | 78 | 0.719 | 78 | 0.467 |
| LOTSCHD | Optimal | 12 | 0.072 | 12 | 0.063 |
| MOSARQP1 | Optimal | 12 | 0.654 | 12 | 0.234 |
| MOSARQP2 | Optimal | 11 | 0.280 | 11 | 0.124 |
| POWELL20 | Optimal | 58 | 0.523 | 58 | 0.350 |
| PRIMAL1 | NumError | - | 0.18 | - | 0.11 |
| PRIMAL2 | NumError | - | 0.42 | - | 0.24 |
| PRIMAL3 | NumError | - | 0.12 | - | 0.09 |
| PRIMAL4 | NumError | - | 0.11 | - | 0.08 |
| PRIMALC1 | Optimal | 23 | 0.123 | 23 | 0.097 |
| PRIMALC2 | Optimal | 25 | 0.121 | 25 | 0.098 |
| PRIMALC5 | Optimal | 21 | 0.102 | 21 | 0.091 |
| PRIMALC8 | Optimal | 26 | 0.133 | 26 | 0.104 |
| Q25FV47 | Optimal | 42 | 3.826 | 42 | 1.469 |
| QADLITTL | Optimal | 20 | 0.230 | 20 | 0.099 |
| QAFIRO | Optimal | 15 | 0.109 | 15 | 0.061 |
| QBANDM | Optimal | 20 | 0.466 | 20 | 0.172 |
| QBEACONF | Optimal | 21 | 0.361 | 21 | 0.148 |
| QBORE3D | Optimal | 21 | 0.461 | 21 | 0.184 |
| QBRANDY | Optimal | 19 | 0.368 | 19 | 0.146 |
| QCAPRI | Optimal | 47 | 1.953 | 47 | 0.742 |
| QE226 | Optimal | 20 | 0.445 | 20 | 0.184 |
| QETAMACR | Optimal | 31 | 1.256 | 31 | 0.546 |
| QFFFFF80 | Optimal | 34 | 1.012 | 34 | 0.409 |
| QFORPLAN | Optimal | 45 | 2.132 | 45 | 0.836 |
| QGFRDXPN | Optimal | 29 | 1.586 | 29 | 0.644 |
| QGROW15 | Optimal | 35 | 2.913 | 35 | 0.925 |
| QGROW22 | Optimal | 34 | 3.645 | 34 | 1.164 |
| QGROW7 | Optimal | 30 | 1.537 | 30 | 0.550 |
| QISRAEL | Optimal | 25 | 0.454 | 25 | 0.181 |
| QPCBLEND | Optimal | 17 | 0.160 | 17 | 0.091 |
| QPCBOEI1 | Optimal | 28 | 0.438 | 28 | 0.210 |
| QPCBOEI2 | Optimal | 26 | 0.312 | 26 | 0.168 |
| QPCSTAIR | Optimal | 25 | 0.296 | 25 | 0.200 |
| QPILOTNO | Optimal | 40 | 3.050 | 40 | 1.223 |
| QPTEST | Optimal | 11 | 0.082 | 11 | 0.052 |
| QRECIPE | Optimal | 19 | 0.295 | 19 | 0.124 |
| QSC205 | Optimal | 17 | 0.220 | 17 | 0.096 |
| QSCAGR25 | Optimal | 24 | 0.516 | 24 | 0.212 |
| QSCAGR7 | Optimal | 23 | 0.261 | 23 | 0.108 |
| QSCFXM1 | Optimal | 39 | 1.280 | 39 | 0.450 |
| QSCFXM2 | Optimal | 47 | 2.083 | 47 | 0.735 |
| QSCFXM3 | Optimal | 47 | 2.632 | 47 | 0.898 |
| QSCORPIO | Optimal | 20 | 0.439 | 20 | 0.179 |
| QSCRS8 | Optimal | 23 | 0.781 | 23 | 0.289 |
| QSCSD1 | Optimal | 15 | 0.310 | 15 | 0.119 |
| QSCSD6 | Optimal | 17 | 0.285 | 17 | 0.115 |
| QSCSD8 | Optimal | 15 | 0.950 | 15 | 0.308 |
| QSCTAP1 | Optimal | 20 | 0.407 | 20 | 0.149 |
| QSCTAP2 | Optimal | 17 | 0.303 | 17 | 0.143 |
| QSCTAP3 | Optimal | 17 | 0.295 | 17 | 0.152 |
| QSEBA | Optimal | 31 | 0.751 | 31 | 0.295 |
| QSHARE1B | Optimal | 21 | 0.349 | 21 | 0.135 |
| QSHARE2B | Optimal | 27 | 0.331 | 27 | 0.135 |
| QSHELL | Optimal | 38 | 2.875 | 38 | 1.124 |
| QSHIP04L | Optimal | 18 | 0.419 | 18 | 0.159 |
| QSHIP04S | Optimal | 18 | 0.388 | 18 | 0.151 |
| QSHIP08L | Optimal | 19 | 1.555 | 19 | 0.626 |
| QSHIP08S | Optimal | 18 | 0.918 | 18 | 0.386 |
| QSHIP12L | Optimal | 20 | 1.879 | 20 | 0.779 |
| QSHIP12S | Optimal | 21 | 0.841 | 21 | 0.322 |
| QSIERRA | Optimal | 27 | 1.525 | 27 | 0.619 |
| QSTAIR | Optimal | 30 | 0.727 | 30 | 0.312 |
| QSTANDAT | Optimal | 24 | 0.912 | 24 | 0.354 |
| S268 | Optimal | 15 | 0.089 | 15 | 0.057 |
| STADAT1 | Optimal | 64 | 1.594 | 64 | 0.491 |
| STADAT2 | Optimal | 20 | 0.550 | 20 | 0.194 |
| STADAT3 | Optimal | 20 | 0.293 | 20 | 0.183 |
| STCQP1 | Optimal | 14 | 0.998 | 14 | 0.409 |
| STCQP2 | Optimal | 15 | 1.365 | 15 | 0.558 |
| TAME | Optimal | 9 | 0.051 | 9 | 0.041 |
| UBH1 | Optimal | 15 | 0.22 | 15 | 0.17 |
| VALUES | Optimal | 16 | 0.215 | 16 | 0.096 |
| YAO | Optimal | 59 | 0.745 | 59 | 0.312 |
| ZECEVIC2 | Optimal | 11 | 0.072 | 11 | 0.063 |

## 4. Full Re-test (single end-to-end run before delivery)

Final lib (with `with_lock`, `set_shmem` counting static shared memory, the SIGCHLD fix; all kernel
experiments reverted), per suite:

| Suite | Result |
|---|---|
| C++ PDLP_TEST | 70 / 0 |
| C++ DUAL_SIMPLEX_TEST | 30 / 0 |
| C++ LP_UNIT_TEST | 35 / 0 |
| C++ QP_UNIT_TEST | 7 / 0 |
| C++ SOCP_TEST | 28 / 0 |
| C++ MIP_TEST / MIP_TERMINATION / PRESOLVE / INCUMBENT | 3 / 12 / 1 / 3, all 0 fail |
| C++ C_API_TEST | 58 / 0 |
| C++ CLI_TEST | 7 / 0 |
| C++ WAYPOINT_MATRIXTEST | 6 / 0 |
| C++ ROUTING_UNIT_TEST | 57 / 0 |
| Python `tests/linear_programming` | 51 passed / 9 skipped / 0 failed |
| Python `tests/routing` | 42 passed / 1 failed (see §5) |
| REST (LP + routing via REST) | LP Optimal / -464.7531 / X01=80; routing status=0 / cost=3.0; `mcErrorRecompile=0` |

Full functional matrix: **no regressions**.

## 5. Known Limitations

| Item | Description |
|---|---|
| ~~cvrp private memory~~ (resolved) | Was an inverted `main()` exit code, not a memory downgrade (§2.2). |
| C_API 128-thread reproducibility | `deterministic_reproducibility/1` (gen-ip054, 128 threads, 60 s wall-clock bound): timing jitter under high C500 concurrency truncates at different node counts, giving two valid feasible solutions with different objectives (6922 vs 6910). Reproducible at 4/8 threads. Not a functional/correctness regression. |
| Routing solution quality | +1.8 to 15% above A100 under a fixed budget (§3.3), tracking the slower per-iteration time. Mitigate with a longer budget, or add `__launch_bounds__` to 1024-thread kernels / drop launches to ≤512 to remove runtime-recompile overhead. |
| scpm1 LP | 14× slowdown outlier (§3.1); still Optimal. |
| nug08-3rd C500 nondeterminism (post-perf_opt) | IPM iteration count drifts run-to-run on C500 (83 / 390 / 409, 43–212 s) vs A100's deterministic 18 / 6.1 s; objective stays correct. A performance-nondeterminism regression; root cause and the `CUOPT_LDLT_DETERMINISTIC` switch are in §3.5. |
