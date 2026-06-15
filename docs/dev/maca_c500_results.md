# cuOpt 在 MetaX C500 (MACA) 上的功能与性能对标结果

> **目标**:在 MetaX C500 / MACA 3.0.x / cu-bridge(warp64)上,使 cuOpt 达到与官方 RAPIDS-free + 自研 LDLᵀ wheel（A100 基线,见 `docs/dev/rapids_removal_mvp_results.md`）一致的功能范围,并完成性能基准对比。
>
> **硬件**:对照基线 = NVIDIA A100;本表 = MetaX C500 + AMD EPYC 7543。
>
> **运行约定**:构建与测试前 `source maca_env.sh`(含 `MACA_DIRECT_DISPATCH=1`),构建目录 `cpp/build_maca`。C500 与 NVIDIA 的通用差异记录于 `skills/cuopt-maca-porting/SKILL.md`。

## 1. 功能范围矩阵

| 功能面 | A100 | C500 | 验证 / 备注 |
|---|---|---|---|
| LP — PDLP | ✅ | ✅ | PDLP_TEST 70 / 0 |
| LP — 对偶单纯形 | ✅ | ✅ | DUAL_SIMPLEX_TEST 30 / 0 |
| LP — Barrier | ✅ | ✅ | LP_UNIT / C_API / CLI barrier 路径通过 |
| MILP | ✅ | ✅ | MIP_TEST 3、MIP_TERMINATION 12、PRESOLVE 1、INCUMBENT 3,均 0 fail |
| QP / SOCP（自研 LDLᵀ） | ✅ | ✅ | QP_UNIT 7 / 0、SOCP 28 / 0 |
| C API | ✅ | ✅ | C_API_TEST 57 / 1；唯一失败为 128 线程时限可复现性,非功能回归(见 §5) |
| cuopt_cli | ✅ | ✅ | CLI_TEST 7 / 0（需 `cuopt_cli` 在 PATH） |
| routing C++ | ✅ | ✅ | ROUTING_UNIT_TEST 57 / 0；修复 span<bool>、自旋锁、cycle-finder（§2.1） |
| 距离引擎 | ✅ | ✅ | WAYPOINT_MATRIXTEST 6 / 0 |
| examples（cvrp/pdptw/service） | ✅ | ✅ | 三例均 SUCCESS;service/pdptw 修过时示例数据;cvrp 修上游 main() 退出码逻辑反置（§2.2） |
| Python LP/MILP/QP/SOCP | ✅ | ✅ | `tests/linear_programming` 51 passed / 9 skip / 0 fail（§2.3） |
| Python routing + distance | ✅ | ✅ | `tests/routing` 42 passed（含 distance 8/8）;修复 cross-move 自旋锁（§2.4） |
| REST server / self-hosted | ✅ | ✅ | LP + routing via REST 端到端通过;一行修 SIGCHLD（§2.6） |
| 性能基准 | ✅ | ✅ | LP/QP 目标值一致、耗时同量级;routing 解质量近基线（§3） |

> 两处非绿（C_API 128 线程可复现性、Python routing 的 numpy 告警顺序）均为非功能/非 MACA 项,详见 §5。

## 2. 关键问题与修复

逐项记录 C500 特有缺陷的 **症状 / 根因 / 修复 / 验证**;通用诊断范式见 c500 skill。

### 2.1 C++ routing：三处 C500 特有内核缺陷

**(a) `span<bool const>` 花括号初始化构造出垃圾 span。** vendored `raft::span`（`cpp/vendor/include/raft/core/span.hpp`）以 `base_{ptr, count}` 初始化 `cuda::std::span`;对 `bool const` 这是收窄列表初始化,老 MACA CCCL 静默编错（`.data()` 为野指针）。cuOpt 每车 `order_match` 即 `span<bool const>`,device 解引用触发 Xnack/ATU 陷阱,级联拖垮整个 ROUTING_UNIT。**修复**:改为圆括号 direct-init `base_(ptr, count)`（各平台均正确）。最小复现 `docs/dev/maca_span_bool_repro.cu`。

**(b) warp 协作自旋锁在 C500 死锁/泄漏。** `acquire_lock`/`release_lock`（`cpp/src/utilities/cuda_helpers.cuh`）依赖 Volta 独立线程调度（ITS）。C500 为 lock-step SIMT（warp64）、无 ITS:同一 warp 内 ≥2 lane 争用同一锁时,赢得 CAS 的 lane 永不到达 `release_lock` → 死锁;探测型哈希插入则永久泄漏锁。表现为 100% util、无 trap、无报错。**修复**:新增 `with_lock`/`with_lock_block`(临界区与释放置于中标 CAS 分支内);探测型 `device_map_t::add` 改为 warp 内 lane 串行化。共修复 5 处多-lane 站点（MIP 各处均单-lane,安全)。

**(c) `insert_graph_nodes_kernel` 越界 = cycle-finder 重构竞态。** cycle-finder `extend_cycle` 多 block 并发读写同一可变的 `d_best.key_ptr[0]`,后续 wave 撕裂读到中间态、误匹配出图中不存在的边,最终插入阶段越界。**修复**:改为双缓冲(读 `key_in`、唯一命中线程写 `key_out`,读写集内存不相交、按构造无同址读写),层间 swap。

验证:`ROUTING_UNIT_TEST` 全套件 **57 / 0**（span 修复前为 5 passed / 105 failed 的级联崩溃)。

### 2.2 examples（cvrp / pdptw / service）

三例均编译通过、三个场景全部求解 SUCCESS。service_team_routing、pdptw_mixed_fleet 的失败为**过时示例数据**(主机端校验,A100 现版同样失败):前者 `break_locations` 非唯一,后者混合车队未按车型注册 cost matrix;均已修。

cvrp_daily_deliveries 此前的"退出码 1"曾被误判为私有内存降级,实为**上游 `main()` 退出码逻辑反置**(初始 OSS 提交 b833f7a1 即有):该例写成 `if (status) { exit_code = 1; }`(status 为 true 即 Success),三场景全成功反而返回 1;同目录 pdptw/service 两例都是正确的 `if (!status)`。此 bug 在 A100 上同样返回 1。**已修**为 `if (!status)`,现退出码 0。

关于私有内存:在系统侧将 `pri_mem_sz` 由只读 16 KB 提高到 24 KB(`sudo modprobe metax pri_mem_sz=24`)后,某 local-search 内核曾需 20 KB/thread 的请求不再触发降级——MXLOG 无 private/spill/降级告警、`trapInfo.*` 无新 trap、三场景均 SUCCESS。即 C500 私有内存硬限可由内核参数放宽,而非代码所致的退出码问题。

### 2.3 Python LP/MILP/QP/SOCP（RAPIDS-free）:全绿

`tests/linear_programming` **51 passed / 9 skip / 0 fail**。修复两处根因:

1. **同进程重复求解 fault**:`csrsort_cusparse`（`problem_helpers.cuh`）的 `cusparseXcsrsort` 内部 mccub 内核在 C500 async 下竞态,fault 后运行时被禁用、级联崩溃。逐步排除（`MACA_LAUNCH_BLOCKING=3` 复跑全过、独立复现真实输入无误、裸 mccub 无 fault）确认为 mcsparse 内部驱动竞态。**修复**:改用 CUB `DeviceSegmentedSort::SortPairs`。
2. **`get_vars()` 返回 inf**:RAPIDS-free Cython 模块以 NVIDIA `libcudart` 链接,`cudaMemcpy`（D2H）在 MACA 设备指针上静默失败（rc=1）、`np.empty` 留垃圾。**修复**:该 TU 经 cu-bridge 编译（`build_lp_modules_rapids_free.sh` 的 `CUOPT_RT_MACA=1`:`pre_make nvcc` + cu-bridge CCCL + `-lruntime_cu -lmcruntime`),`cudaMemcpy` 在编译期改写为 `wcudaMemcpy`、走 MACA 运行时。

### 2.4 Python routing + distance（mcdf）

**mcdf 安装**(C500 的 cudf/rmm/cupy/numba):`pip install --no-deps` 四个 cp310 wheel（`mcdf`→cudf、`mcpy`→cupy、`rmmx`、`numbax`),并补纯 Python 依赖（`fastrlock`、`llvmlite==0.39.1`、`pandas==1.5.3`、`pyarrow==10.0.1`、`protobuf==4.21.12`）。两处适配:cupy 在线 JIT 需 `CUDA_PATH=/opt/maca/tools/cu-bridge`(否则取到 NVIDIA `cuda_fp16.h`,mxcc 无法解析);cuOpt 编译进 `.so` 的 `from numba import cuda` 需 `numba`→`numbax` 垫片。一键脚本 `python/cuopt/setup_maca_rapids.sh`。

**cross-move 自旋锁跨 block 死锁(已修)**:`test_re_routing` 挂起。gdb 采样定位到 `populate_cross_list_kernel` 的 `route_pair_locks` 用裸 `acquire_lock`/`release_lock`;虽同 block 内每 lane 锁不同,但多 block 共享 `first_route`、**跨 block 争用同一锁**,在 C500 lock-step 下死锁(随问题规模增长才触发,故早先漏判)。**修复**:改用 `with_lock`(同 §2.1b)。验证:`test_re_routing` 11/11、`tests/routing` 42 passed。

### 2.5 共享内存 64 KB 上限加固

C500 限制的是**静态+动态**共享内存总量 64 KB,而 `set_shmem_of_kernel` 原仅校验动态部分:带静态 `__shared__` 的核可"动态 ≤64 KB 通过校验"却"静态+动态 >64 KB 启动失败"→ 崩溃（`find_all_squeeze_pos` 在 ~1000+ 单路由触发)。**修复**:`set_shmem_of_kernel` 计入 `cudaFuncGetAttributes().sharedSizeBytes`,超 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 即返回 false → 调用方干净跳过/抛 OOM。修复后超限核绝不带不足内存启动,**无静默错算**;结果为 SUCCESS(有效解)或明确 ERROR/INFEASIBLE。普通多车 VRP(路由短)不触发。

### 2.6 REST server:一行修 `SIGCHLD=SIG_DFL`

LP via REST 端到端通过（afiro Optimal / -464.7531 / X01=80）。routing via REST 曾因 fork 的 solver 子进程刷 `mcErrorRecompile` 而失败。**根因链**:C500 对「launch block size 超过该核编译期 `max_block_size`(默认 512)」的核做运行期重编译（fork+exec `mxcc` 子进程并 `wait()`);而 solver worker（`solver.py::process_async_solve`）设 `SIGCHLD=SIG_IGN`,在 Linux 上致 `wait()` 返 ECHILD、`mxcc` build 失败、launch 返回 `mcErrorRecompile`。直连(无此 handler)重编译正常,故仅在 server 复现。**修复**:改为 `SIG_DFL`(仍忽略 SIGCHLD 但 `wait()` 可用),无需改任何 kernel。验证:`mcErrorRecompile=0`,routing via REST status=0、cost 3.0/304/8(与直连一致)。

> 部署提示:本机经 HTTP 代理,打本地端口须 `no_proxy` / `--noproxy`(否则 502);服务器以 `nvidia-smi` 枚举 GPU(本机另有一块 A100,装饰性),实际 solver 进程跑在 C500（`mx-smi` 确认)。

## 3. 性能基准(C500 vs A100)

基线 = `rapids_removal_mvp_results.md` §5 的 A100 实测(RAPIDS-free + 自研 LDLᵀ,即官方 cuDSS-free wheel 等价)。

### 3.1 LP（Mittelmann 子集,Barrier `--method 3 --time-limit 600`）

全部 Optimal,目标值与基线一致。

| 实例 | A100 耗时 | A100 目标值 | C500 耗时 | C500/A100 |
|---|---|---|---|---|
| afiro | 0.09 s | -464.753135 | 0.08 s | 0.9× |
| graph40-40 | 0.99 s | -300.000464 | 1.14 s | 1.15× |
| qap15 | 1.41 s | 1041.00046 | 2.74 s | 1.94× |
| nug08-3rd | 6.23 s | 214.000642 | 11.35 s | 1.82× |
| woodlands09 | 5.42 s | ≈0 | 10.04 s | 1.85× |
| scpm1 | 4.20 s | 414.152071 | 60.25 s | 14.3× |

中型实例 1.1–1.9×(与基线对 cuDSS 的 1.3–1.7× 同量级,系自研 LDLᵀ 在 C500 上的因子分解差距)。scpm1 离群 14×(疑该填充模式下 LDLᵀ 因子分解的病态慢路径,不影响正确性,仍 Optimal)。

### 3.2 QP（Maros-Mészáros 子集,Barrier `--method 3 --time-limit 180`）

全部 Optimal,目标值与基线**逐位一致**,耗时 0.05–0.11 s（基线 0.04–0.09 s）。

| 问题 | A100 目标值 | C500 目标值 | C500 耗时 |
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

### 3.3 Routing（CVRPLIB Set-X / Solomon / Homberger,固定时间预算）

基线 = `rapids_removal_mvp_results.md` §5.4;驱动 `docs/dev/route_bench.py`。全部 SUCCESS、车辆数合法、TW/容量约束满足。

| 实例(规模,预算) | A100 基线（2 次） | 文献 BKS | C500 cost | C500 车辆 | C500/基线 |
|---|---|---|---|---|---|
| X-n101-k25（101,30 s） | 31573 / 32753 | 27591 | 36313 | 25 | +12~15% |
| X-n439-k37（439,60 s） | 36454 / 36470 | 36391 | 37138 | 37 | +1.8% |
| X-n1001-k43（1001,120 s） | 73709 / 74042 | 72355 | 77209 | 43 | +4.5% |
| C101（Solomon 100+TW,30 s） | 828.94 / 828.94 | 828.94 | 848.94 | 11 | +2.4% |
| C1_10_1（Homberger 1000+TW,120 s） | 42478.95 / 42478.95 | ≈42444 | 46767 | 147 | +10% |

固定预算下解质量较基线高 **+1.8~15%**(略逊),与 §3.1 一致:C500 每迭代慢约 1.1–1.9×,同一 wall-clock 预算内启发式迭代更少。差距在最细粒度/最大实例上最明显,中型接近基线。非正确性问题。

## 4. 全量复测(交付前一次性重跑)

最终 lib（含 `with_lock`、`set_shmem` 静态共享计入、SIGCHLD 修复;kernel 实验已全部回退）逐套件:

| 套件 | 结果 |
|---|---|
| C++ PDLP_TEST | 70 / 0 |
| C++ DUAL_SIMPLEX_TEST | 30 / 0 |
| C++ LP_UNIT_TEST | 35 / 0 |
| C++ QP_UNIT_TEST | 7 / 0 |
| C++ SOCP_TEST | 28 / 0 |
| C++ MIP_TEST / MIP_TERMINATION / PRESOLVE / INCUMBENT | 3 / 12 / 1 / 3,均 0 fail |
| C++ C_API_TEST | 57 / 1（唯一失败见 §5） |
| C++ CLI_TEST | 7 / 0 |
| C++ WAYPOINT_MATRIXTEST | 6 / 0 |
| C++ ROUTING_UNIT_TEST | 57 / 0 |
| Python `tests/linear_programming` | 51 passed / 9 skip / 0 fail |
| Python `tests/routing` | 42 passed / 1 fail（见 §5） |
| REST（LP + routing via REST） | LP Optimal/-464.7531/X01=80;routing status=0/cost=3.0;`mcErrorRecompile=0` |

全矩阵功能**无回归**。

## 5. 已知限制

| 项 | 说明 |
|---|---|
| ~~cvrp 私有内存~~（已解决） | 此前判为私有内存降级,实为上游 `main()` 退出码逻辑反置(已修,见 §2.2)。私有内存侧:`sudo modprobe metax pri_mem_sz=24`(>20 KB)后某 local-search 内核 20 KB/thread 请求不再降级,无告警、无新 trap、三场景 SUCCESS、退出码 0。 |
| C_API 128 线程可复现性 | `deterministic_reproducibility/1`（gen-ip054,128 线程,受 60 s wall-clock 约束）在 C500 高并发下计时抖动致截断落在不同 node 数,两次均为**合法可行解**但目标值不同（6922 vs 6910）;4/8 线程可复现。非功能/正确性回归。 |
| routing 解质量 | 固定预算下较 A100 高 +1.8~15%（§3.3），随 C500 单迭代变慢。可用更长预算,或给 1024-thread 核加 `__launch_bounds__` / 降 launch ≤512 以消除运行期重编译开销来对齐。 |
| scpm1 LP | 14× 慢离群（§3.1),仍 Optimal。 |
| Python routing numpy 告警顺序 | `test_type_casting_warnings` 在 numpy 1.26 下失败（`find_common_type` deprecation 告警先于 cast 告警）,A100 同 numpy 亦失败,非 MACA。 |
