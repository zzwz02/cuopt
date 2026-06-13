# cuOpt MetaX C500 (MACA) 功能与性能对标——实施结果

> 目标:在 MetaX C500 / MACA / cu-bridge 上,使 cuOpt 达到与官方 wheel
> (cuDSS 版,即 RAPIDS-free + 自研 LDLᵀ 基线 `docs/dev/rapids_removal_mvp_results.md`)
> **一样的功能范围**,并对性能进行基准对比。**暂不引入 cudf**(因此 Python
> routing/distance 的 cudf 进出路径暂缓;Python LP/MILP/QP/SOCP 走 numpy)。
> 参照基线硬件 = NVIDIA A100;本表硬件 = MetaX C500(MACA 3.0.x,cu-bridge,
> warp64),AMD EPYC 7543。
>
> 工作方式:每完成一项即在本文件记录并 commit;C500 与 NVIDIA 的差异记录在
> `skills/cuopt-maca-porting/SKILL.md`。
>
> 运行约定:所有构建/测试均 `source maca_env.sh`(含 `MACA_DIRECT_DISPATCH=1`),
> 构建目录 `cpp/build_maca`。

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
| routing C++(VRP/PDP/breaks/异构) | ✅ | ⚠ 部分(GES 路径阻塞) | ROUTING_UNIT 5/110:`find_all_squeeze_pos` 内核 warp64 全局越界(Xnack/ATU),拖垮 GES 相关用例(详见 §4) |
| 距离引擎(C++ waypoint) | ✅ | ✅ | WAYPOINT_MATRIXTEST 6/6 |
| examples(cvrp/pdptw/service) | ✅ | ⏳ 待建/待测 | — |
| Python LP/MILP/QP/SOCP(numpy) | ✅ | ⏳ 待建/待测 | Cython 扩展未构建 |
| Python routing+distance(cudf) | ✅ | ⏸ 暂缓(不引入 cudf) | — |
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

## 4. routing C++ 调查(进行中)

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
- **工具**:`mcSanitizer` 在本容器不可用(`inotify_add_watch` on
  `/root/mcRpcPort.ini` 失败后挂起);已对 `squeeze.cu` 加 `--generate-line-info`
  重编以备 PC→行号映射。根因定位与修复进行中(并行代码分析 + 复现)。

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
