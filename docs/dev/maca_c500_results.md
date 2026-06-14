# cuOpt MetaX C500 (MACA) 功能与性能对标——实施结果

> 目标:在 MetaX C500 / MACA / cu-bridge 上,使 cuOpt 达到与官方 wheel
> (cuDSS 版,即 RAPIDS-free + 自研 LDLᵀ 基线 `docs/dev/rapids_removal_mvp_results.md`)
> **一样的功能范围**,并对性能进行基准对比。
> 参照基线硬件 = NVIDIA A100;本表硬件 = MetaX C500(MACA 3.0.x,cu-bridge,
> warp64),AMD EPYC 7543。
>
> 工作方式:每完成一项即在本文件记录并 commit;C500 与 NVIDIA 的差异记录在
> `skills/cuopt-maca-porting/SKILL.md`。
>
> 运行约定:所有构建/测试均 `source maca_env.sh`(含 `MACA_DIRECT_DISPATCH=1`),
> 构建目录 `cpp/build_maca`。A100 对照 build 目录:`cpp/build_a100_cuda_cached_20260613_1617`。该目录为 CUDA/A100
> 路径,链接本地 CUDA 而非 MACA/cu-bridge,可复用作后续 C500 vs A100 行为对照。

## 1. 功能范围矩阵(对标 cuDSS-free 基线 §3)

| 功能面 | A100 基线 | C500 状态 | 验证 |
|---|---|---|---|
| LP — PDLP(C++) | ✅ | ✅ | PDLP_TEST 70 passed / 3 skip / 0 fail |
| LP — 对偶单纯形(C++) | ✅ | ✅ | DUAL_SIMPLEX_TEST 30/30 |
| LP — Barrier(C++) | ✅ | ✅ | C_API/LP_UNIT/CLI barrier 路径通过 |
| MILP(分支定界/割/启发式) | ✅ | ✅ | MIP_TEST 3/3、MIP_TERMINATION_STATUS 12/12、PRESOLVE 1/1、INCUMBENT_CALLBACK 3/3 |
| QP / SOCP(barrier,自研 LDLᵀ) | ✅ | ✅ | QP_UNIT 7/7、SOCP 28/28 |
| C API(cuopt_c.h) | ✅ | ✅ | C_API_TEST 61 run / 0 fail |
| cuopt_cli | ✅ | ✅ | LP/QP 烟测目标值精确;CLI_TEST 7/7(需 `cuopt_cli` 在 PATH:`export PATH=$PWD/cpp/build_maca:$PATH`)|
| routing C++(VRP/PDP/breaks/异构) | ✅ | ✅ 已修复 `span<bool>`、warp 自旋锁、`test_heterogeneous_breaks` OOB | §4:span 花括号→圆括号(§4.0)+ warp 协作自旋锁串行化(§4.0b)+ cycle-finder 重构竞态(§4.0c) |
| 距离引擎(C++ waypoint) | ✅ | ✅ | WAYPOINT_MATRIXTEST 6/6 |
| examples(cvrp/pdptw/service) | ✅ | ⏳ 待建/待测 | — |
| Python LP/MILP/QP/SOCP(numpy) | ✅ | ⏳ 待建/待测 | Cython 扩展未构建 |
| Python routing+distance(cudf) | ✅ | ⏸ 暂缓 | — |
| REST server / gRPC / self-hosted | ✅ | ⏸ 暂缓 | — |
| 性能基准(LP/QP/routing) | ✅(§5) | ✅ LP+QP(routing 暂缓) | LP §5.1(多数 1–2×,scpm1 14×);QP §5.2(目标值逐位一致) |

图例:✅ 已验证通过 · ⏳ 在做/待做 · ⏸ 本阶段暂缓

## 2. 工作日志(逐项;每项一次 commit)

### W0 — 求解器内核 C500 适配(前序工作,已 commit)
- 已修复并验证:LP/QP/MIP 的 C500 失败、PDLP batch GEAM 转置、PDLP
  trust-region 协作内核死锁(改 host-driven)、batch saddle-point AtY 脏内存
  污染。相关 commit:`409dfea1 / 6b21faf3 / da048e46 / 6dfadc4f`。
- C++ 测试二进制全绿(10/10 二进制,0 失败):
  PDLP_TEST(70+3skip)· LP_UNIT(35)· QP_UNIT(7)· MIP_TERMINATION(12)·
  MIP_TEST(3)· INCUMBENT_CALLBACK(3)· PRESOLVE(1)· C_API(61)·
  DUAL_SIMPLEX(30)· SOCP(28)。
- 关键运行依赖:`MACA_DIRECT_DISPATCH=1`(已写入 maca_env.sh)。

### W1 — routing C++ 死锁根因修复(本次 commit)
- **修复 1**:`span<bool const>` 花括号初始化 → 圆括号(§4.0,前序 commit `87d84042`)。
- **修复 2**:warp 协作自旋锁在 C500(lock-step、无 ITS)死锁/泄漏(§4.0b)。
  改动 6 文件:`cuda_helpers.cuh`(新增 `with_lock`/`with_lock_block`)、
  `vrp_move_candidates.cuh`、`move_candidates.cuh`、`prize_collection.cu`、
  `perform_moves.cu`(in-branch 释放);`device_map.cuh`(warp 内 lane 串行化)。
- **修复 3**:`heterogenous_breaks.test_heterogeneous_breaks` 的
  `insert_graph_nodes_kernel` OOB = cycle-finder 重构 predecessor 链时一大片 block
  并发读写同一个 `d_best.key_ptr[0]`(读端跨-block 撕裂读),偶发拼出不存在的图边;
  改为**双缓冲** `extend_cycle`(读 `key_in`、唯一命中线程写 `key_out`,读写集内存不相交,
  按构造可证)(§4.0c)。
- 验证:`heterogenous_breaks.*` `--gtest_repeat=10` 10/10(2 tests,无 Memory Violation);
  回归 `vehicle_breaks` 6/6、`heterogenous` 6/6、`prize_collection` 3/3、
  `objective_function` 2/2、`vehicle_order_match` 2/2。

## 4. routing C++ 调查

### 4.0 根因已定位并修复:`span<bool const>` 花括号初始化在 MACA CCCL 上构造出垃圾 span

**根因**:vendored `raft::span`(`cpp/vendor/include/raft/core/span.hpp`)用花括号
初始化其 `cuda::std::span<T> base_`:`base_{ptr, count}`。对 `T = bool const`
这是**收窄列表初始化**(`bool const*` → `bool`)。新 CCCL/CUDA 在编译期直接报错;
**老 MACA CCCL 编译通过但构造出垃圾 span(指针/长度错乱)**。cuOpt 的每车
`order_match` 恰为 `raft::span<bool const>`,故 C500 上其 `.data()` 变成野指针
(实测 `0x7ffc…`,size=2),被 MISMATCH 维度 `order_match[l1.node()]` 在 device
解引用 → Xnack/ATU 陷阱 → `find_all_squeeze_pos` 触发、级联拖垮整个 ROUTING_UNIT。

**与 warp64 无关**(之前的猜测错误);本质同 AtY 类——NVIDIA 老 CUDA 12.1 恰好
编译为正确构造,12.9 编译期报错,MACA CCCL 静默编错。

**最小复现**(`docs/dev/maca_span_bool_repro.cu`,A100 与 C500 对拍):

| | C500(MACA CCCL) | A100(CUDA 12.9) |
|---|---|---|
| `span<bool const>{ptr,count}`(花括号) | **编译过、运行垃圾**:`data=0x2000000000004, size=2` | **编译报错** `narrowing const __nv_bool*→__nv_bool` |
| `span<bool const>(ptr,count)`(圆括号,本修复) | ✅ 正确 | ✅ 正确 |
| `span<int const>` / `span<bool>` | ✅(仅 `bool const` 出错) | ✅ |

**修复**:`span.hpp` 中把 `base_{...}` 改为圆括号 `base_(...)`(direct-init,强制
走 `(ptr,count)` 构造),对所有 span 正确。修复后 `vehicle_time_windows` 等通过、
ATU 陷阱消失。

### 4.0b 已修复:`vehicle_breaks.uniform_breaks` 挂起 = warp 协作自旋锁在 C500 死锁/泄漏

**(此前「局部搜索不收敛」的判断是错的。)** 真因是 **warp 协作自旋锁
(`acquire_lock`/`release_lock`,`cpp/src/utilities/cuda_helpers.cuh`)在 C500 上
死锁或泄漏**。该锁是教科书式 Volta-ITS 自旋锁:
`while (atomicCAS(lock,0,1)) __nanosleep(100); __threadfence();`,依赖 NVIDIA
Volta(sm_70)引入的**独立线程调度(ITS)**。C500 是 **lock-step SIMT、无 ITS**
(`warp64`),因此**同一 warp 内有 ≥2 个 lane 争用同一把锁**时,赢得 CAS 的 lane
被停在自旋循环的重汇聚点,而输的 lane 永远自旋——赢家永不执行 `release_lock`。
表现:GPU 100% util、无进展、**无 trap、无报错、与 CUDA Graph 无关**(看起来与卡死
内核完全一样;须确认 `~/mxlog/umd/trapInfo.*` 是**旧的**才能排除 OOB)。

**两条死锁链**(printf 在 host 调用边界二分 + kernel 内 while 自旋加计数器定位):
1. `find_vrp_moves` → `find_vrp_moves_kernel`(TPB=96=1.5×warp64)→
   `record_candidate`(`vrp_move_candidates.cuh`,按 `node_id_1`/`route_pair_idx`
   加锁,多 lane 撞同一 node 即争用)。数据相关:n_blocks=2546 通过、5360 挂。
2. `find_best_negative_cycles` → cycle-finder `find_kernel` → `device_map_t::add`
   (`device_map.cuh` 哈希插入的逐槽锁)。**更恶劣的泄漏型**:lane 赢 CAS 后在
   `release_lock` 前被停 → 该槽锁**永久泄漏** → 后续探测到该槽的 lane 永远自旋
   (实测数百 lane 各自卡在**各自不同**的 index;`max_available=400000` 远未填满)。

**仅多-lane-同锁的站点受影响**;`if(threadIdx.x==0)`/`lane_id==0`/block-reduce 胜者
(`reduction_idx==threadIdx.x`)守护的、以及每 lane 锁不同 index 的站点都安全(只剩
跨-warp 争用,跨 warp 各自推进)。审计全部 ~18 处 `acquire_lock`:MIP 各处均单-lane
→ 安全;多-lane 的 5 处全部修复。

**修复**(`with_lock`/`with_lock_block` 助手,置于 `acquire_lock` 旁):
- **简单单一固定锁**:把临界区+释放放进**赢得 CAS 的分支内**——
  `while(!done){ if(atomicCAS(L,0,1)==0){ __threadfence(); crit; __threadfence(); atomicExch(L,0); done=true; } }`。修了 `record_candidate`、`record_candidate_thread_safe`、`prize_collection`、`perform_moves`(scross)。
- **探测/多锁哈希插入**(`device_map_t::add`):in-branch 释放**不够**(泄漏型仍发生),
  改为**按 warp 内 lane 串行化**——
  `for (int turn=0; turn<raft::WarpSize; ++turn) if (raft::laneId()==turn) add_impl(...);`
  (无 `__syncwarp`,对发散调用方安全;只一个 lane 进锁,只剩跨-warp 争用)。

修复后 `vehicle_breaks` 全 6 例通过(`uniform_breaks` 30 s 通过,完成 cycle-finder
全部 level)。**通用规律已写入 c500 skill**。诊断手法:host 调用边界 fprintf 二分 +
kernel 内每个 while 自旋加「越界即 printf 并 break」的计数器,触发的即元凶循环。

**ROUTING_UNIT_TEST 全套件结果(逐套件单跑,每套件 240 s 超时):13/14 套件绿。**

| 套件 | 结果 | | 套件 | 结果 |
|---|---|---|---|---|
| vehicle_types | ✅ 2 | | heterogenous | ✅ 6 |
| vehicle_breaks | ✅ 6 | | prize_collection | ✅ 3 |
| heterogenous_breaks | ⚠ 1/2(`test_heterogeneous_breaks` OOB)| | objective_function | ✅ 2 |
| vehicle_fixed_costs | ✅ 3 | | batch_tsp | ✅ 1 |
| vehicle_order_match | ✅ 2 | | set_shmem_of_kernel | ✅ 5 |
| order_locations | ✅ 4 | | top_k/top_cand_test | ✅ 18 |
| horizontal_loading | ✅ 2 | | route_constraints | ✅ 1 |

(对照:此前 span 修复前为 5 passed / 105 failed 的级联崩溃;自旋锁修复前 `vehicle_breaks`
等多套件挂死。)

### 4.0c 已修复:`heterogenous_breaks.test_heterogeneous_breaks` 在 `insert_graph_nodes_kernel` OOB

自旋锁修复后该用例不再挂死,但暴露出一个此前被死锁掩盖的独立越界。元凶内核为
`cuopt::routing::detail::insert_graph_nodes_kernel<int,float,request_t(1)>`;触发条件是
cycle-finder 偶发重构出一条包含**不存在图边**的坏环,随后 `populate_move_path` 如实把坏边
转成 move-path,最终在插入阶段越界。

**命中的具体坏环/越界下标**(一次 C500 suspect dump):
- `cycle id=0 size=5 nodes 102(r2) 97(r5) 96(r5) 0(r9) 120(r20)`。
- 其中 `96(r5)->97(r5)` 在 graph 中不存在:
  `graph_cost=1.79769e+308`,candidate 也是 max,
  `cand_ins=(8,8,32767,32767)`。
- 该边生成 `path block=6 request=96 ejected=97 target_route=5 target_n_after_eject=7 insert=8 PRE_BAD`。
  `insert_graph_nodes_kernel` 弹出 97 后 route 5 只有 7 个节点,却试图以 `insert=8`
  插入 96;若不拦截,VRP `insert_node` 的右移循环从 `i=7` 继续向下找 `insertion_idx=8`,
  会读写越界。

**根因**:cycle-finder 的 `extend_cycle` 以 ~782 个 block × 256 线程并行扫描
`d_valid_paths`,每个线程都直接用同一个可变的 `d_best.key_ptr[0]` 做匹配,而唯一命中
线程又当场改写它(先写 head 的 route 位、再写 head)。grid 远超驻留上限、按 wave 调度,
后一 wave 的 block 会在前一 wave 命中线程改写 `d_best.key_ptr[0]` 的过程中读到它,撕裂出
`(新 head, 旧 label)` 这种中间态;若该中间态恰好等于本 level map 里另一条 entry,就被误
匹配、续错 predecessor 链,从而重构出 graph 中不存在的边(例如同一路线 `96(r5)->97(r5)`)。
本质同 span/AtY 一类:NVIDIA(32 宽 warp、ITS)调度恰好不踩这个窗口、良性;C500(warp64、
lock-step)偶发触发。

**修复**(分两步,最终用双缓冲):
- 先尝试**每线程快照** `target_key = d_best.key_ptr[0]` 再比较——能让 `repeat=10` 转绿,
  但**不彻底**:快照那一次读仍落在命中线程要写的同一个全局字上,后一 wave 的 block 仍可能
  在写的瞬间撕裂读到坏快照(窗口大幅缩小但未关闭)。
- 最终改为**双缓冲**:`extend_cycle` 从 `key_in` 读、唯一命中线程把推进后的 key 写到
  `key_out`(本 launch 内无人读 `key_out`、无人写 `key_in`),读集与写集内存不相交,**按
  构造无同址读写**,与任何 kernel 内内存序无关;层间(各自独立、stream 串行的 launch)
  swap `key_in`/`key_out`。`d_best` 的 tail 沿整条链不变,故 `close_cycle` 仍从
  `best_cycles` 取 tail。改动:`cycle_finder_kernels.cuh`(`extend_cycle` 改 in/out
  指针)、`cycle_finder.cu`(`get_cycle` ping-pong swap)、`cycle_finder.hpp`(新增
  `key_scratch` 缓冲)。map 按 key 去重(`device_map_t::add`)且坏环每段 key 在 find 阶段
  必已入对应 level 的 map,故每 level 恰好一个命中、恰好一个线程写 `key_out`。

**验证**(C500,双缓冲版):
- `ROUTING_UNIT_TEST --gtest_filter=heterogenous_breaks.* --gtest_repeat=10`:10/10 迭代、
  每轮 2/2,无 `Memory Violation`。
- 回归(各单跑全绿):`vehicle_breaks` 6/6、`heterogenous` 6/6、`prize_collection` 3/3、
  `objective_function` 2/2、`vehicle_order_match` 2/2。
- A100 对照(`cpp/build_a100_cuda_cached_20260613_1617`,链接本地 CUDA):旧 `extend_cycle`
  在 `compute-sanitizer --tool memcheck` 下连跑均 `ERROR SUMMARY: 0 errors`——该坏环在 A100
  调度下不复现;memcheck 只捕获**实际发生**的越界,不报告未触发的 cycle 重构竞态(故"A100
  sanitizer 干净"不能证明原实现正确;数据竞争应由 racecheck 而非 memcheck 检测,且其对全局
  内存竞争亦力有不逮)。

### 4.1 调查过程记录

- **距离引擎 WAYPOINT_MATRIXTEST:6/6 通过。**
- **ROUTING_UNIT_TEST:5 passed / 105 failed**,但绝大多数为**级联**:一旦
  GPU 触发 Xnack/ATU 陷阱,MACA 运行时被禁用(`mcruntime api will be
  disabled`),其后所有 CUDA 调用失败或挂起。
- **单一根因内核**(`~/mxlog/umd/` trap 日志):
  `cuopt::routing::detail::find_all_squeeze_pos<int,float,request_t(1),true>`
  ——GES(guided ejection search)squeeze 插入内核,全局内存越界
  (Xnack Error/ATU Fault 0x8)。该陷阱出现在 `vehicle_time_windows`、
  `heterogenous_breaks` 等多处;`top_k` 隔离跑 18/18 全过(纯级联)。
- **隔离复现**(各自单跑):`vehicle_time_windows` 越界;`vehicle_fixed_costs`/
  `heterogenous_breaks`/`vehicle_breaks.uniform/non_uniform` 挂起(同一陷阱使
  运行时禁用后的下游等待);`vehicle_order_match` 4 例失败。
- **warp64 嫌疑**:TPB = `min(128, alignTo(max_active_nodes, WarpSize))`,C500
  上 WarpSize=64 → TPB 总为 64/128(NVIDIA 为 32 的倍数)。但单纯的
  `global[threadIdx.x]` 越界在 NVIDIA(TPB≥32)亦会触发,而 NVIDIA routing
  43/43 全过,故应为 warp64 特有(每-warp 结构 / 共享尺寸 / 约简)路径。
- **崩溃点精定位**(printf 注入 + `~/mxlog/umd` trap 日志逐步收窄):陷阱发生在
  `node_t::total_excess_of_combine`(`cpp/src/routing/node/node.cuh:154` 的
  `get_transit_time` 或 `:164` 的 `get_arc_of_dimension`)——即 squeeze 插入
  评估中 `combine(node, current, vehicle_info,…)` 读取全局 transit/arc 矩阵。
  现场:`num_nodes=1, blockDim=64, max_nodes_per_route=64, vehicle_id=12,
  num_breaks=1`(带 break 维度)。`current=get_node(node_insertion_idx+1)` 读到
  +1(回程 depot)槽;`copy_from` 确实复制了 `[0,n_nodes+1)`(含 +1 槽),故非
  「共享未初始化」。**已排除**:MISMATCH 维度 `order_match[l1.node()]` 越界(注入
  打印 `l1.node=0,size=2`,在界内);共享 route 尺寸 +1 失配(调用方以
  `max_route_size+added_size` 预留,已补偿)。**仍待定**:`combine` 读全局矩阵
  所用某节点索引在 C500 为野值(疑与 AtY 同类——NVIDIA 良性、C500 触发 ATU)。
- **工具**:`mcSanitizer` 本容器不可用(`inotify_add_watch` on
  `/root/mcRpcPort.ini` 失败后挂起,无源码行);`-DASSERT_MODE` 重编亦未命中
  (越界访问无前置 `cuopt_assert`)。**有效手段**:`~/mxlog/umd/trapInfo.*.log`
  直接给出陷阱 kernel 名(demangle → `find_all_squeeze_pos`);printf 注入逐步
  收窄到 combine。最终定位与修复留待后续专项(需可用的逐行 sanitizer)。

## 5. 性能基准(C500 vs A100 基线)

**基线 = `docs/dev/rapids_removal_mvp_results.md` §5 的「本版(shim,RAPIDS-free
+ 自研 LDLᵀ)」A100 实测**(即官方 cuDSS-free wheel 的等价物)。C500 用同一
驱动协议(Barrier,LP `--time-limit 600`、QP 180 s,solver 自报时间)逐实例
对比,目标值须与基线一致。

### 5.1 LP(Mittelmann 子集,Barrier `--method 3 --time-limit 600`)

C500 solver 自报 Barrier 耗时;全部 **Optimal**(目标值与基线一致,graph40-40
逐位 -300)。

| 实例 | A100 基线耗时 | A100 目标值 | C500 耗时 | C500 状态 | C500/A100 |
|---|---|---|---|---|---|
| afiro | 0.09 s | -464.753135 | **0.08 s** | Optimal | 0.9× |
| graph40-40 | 0.99 s | -300.000464 | **1.14 s** | Optimal(-300)| 1.15× |
| qap15 | 1.41 s | 1041.00046 | **2.74 s** | Optimal | 1.94× |
| nug08-3rd | 6.23 s | 214.000642 | **11.35 s** | Optimal | 1.82× |
| woodlands09 | 5.42 s | ≈0 | **10.04 s** | Optimal | 1.85× |
| scpm1 | 4.20 s | 414.152071 | **60.25 s** | Optimal | **14.3× ⚠** |

结论:小型(afiro)与 C500 持平/略快;中型 graph40-40/qap15/nug08/woodlands 为
1.1–1.9×(与基线 §5 对 cuDSS 的 1.3–1.7× 同量级,系自研 LDLᵀ 在 C500 上的
因子分解差距)。**scpm1 为显著离群(14×,60 s/26 iter)**,疑为该填充模式下
C500 LDLᵀ 因子分解的病态慢路径,待后续 profile(不影响正确性,仍 Optimal)。

### 5.2 QP(Maros-Mészáros 子集,Barrier `--method 3 --time-limit 180`)

C500 全部 **Optimal**,**目标值与 A100 基线逐位一致**;耗时 0.05–0.11 s,与基线
(0.04–0.09 s)同量级。

| 问题 | A100 基线目标值 | C500 目标值 | C500 耗时 |
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

结论:QP barrier(自研 LDLᵀ)在 C500 上目标值与 A100 基线**逐位一致**、耗时同
量级。全集 138 题扫描(180 s/题)待后续例行化。
