# cuDSS 替换:功能/性能基准对比

对比对象:

- **ours** — 本分支(自研 level 调度 LDLᵀ,系统 CUDA 12.1 工具链构建的
  `cpp/build/cuopt_cli`);
- **baseline** — 官方发布的 cuOpt 26.06 wheel(`libcuopt-cu12==26.6.0`,
  内部使用 **cuDSS 0.7.1**),pip 安装于独立 venv。

硬件:NVIDIA A100-PCIE-40GB,驱动 CUDA 12.9;AMD EPYC 7543(64C/128T)。
方法:`--method 3`=Barrier(受本工作影响的路径)、`2`=Dual Simplex、
`1`=PDLP(后两者代码未变,作健全性对照)。`--time-limit 600`(QPS 180s)。
`solver_t` 为求解器自报时间(秒),`wall` 为含解析/初始化的进程总耗时。

## LP(Mittelmann/PDLP 测试集子集)

| 实例 | 规模特征 | Barrier ours | Barrier baseline (cuDSS) | 目标一致性 |
|---|---|---|---|---|
| afiro | 14×32 | Optimal 0.09s | Optimal 0.14s | -464.753135 双方一致(12 次迭代均相同) |
| graph40-40 | 因子易并行 | Optimal 1.75s | Optimal 0.90s | -300.000464 双方逐位一致 |
| qap15 | 因子填充重 | TimeLimit 600s | Optimal 1.09s | — |
| nug08-3rd | 因子填充重 | 超时(>1500s 强杀) | Optimal 3.74s | (m2 双方 214.000 一致) |
| woodlands09 | 114k 约束,3401 层消去树 | TimeLimit(残差稳定收敛中) | Optimal 4.13s | — |
| scpm1 | 大规模 | TimeLimit 838s | Optimal 3.71s | — |

对照组(不受影响路径,验证无意外回归):

- Dual Simplex(m2):六个实例双方状态/目标/耗时全部一致
  (如 nug08-3rd:ours 272.1s vs baseline 265.8s,目标同为 214.000000;
  qap15/graph40-40/woodlands09/scpm1 双方同样 600s 时限)。
- PDLP(m1):全部 Optimal,秒级。

## QP(QP_Test + Maros-Mészáros 最小 10 题)

12/12 双方均 Optimal,目标值一致(差异 ≤1e-9,且 HS268/S268/HS51 上
ours 残差略优);耗时 ours 0.04–0.09s vs baseline 0.07–0.14s
(小问题上 cuDSS 初始化开销占主导,ours 反而略快)。

| 问题 | ours | baseline | 已知最优(文献) |
|---|---|---|---|
| QP_Test_1 | -99.9600000 | -99.9600000 | — |
| HS35 | 0.111111115 | 0.111111115 | 1/9 ≈ 0.111111 |
| HS51 | 0.000000000 | 1.78e-15 | 0 |
| HS52 | 5.32664756 | 5.32664756 | 5.326648 |
| HS53 | 4.09302326 | 4.09302326 | 4.093023 |
| HS76 | -4.68181817 | -4.68181817 | -4.681818 |
| HS268/S268 | 3.84e-09 | 3.86e-09 | 0 |
| HS35MOD | 0.250000001 | 0.250000001 | 0.25 |
| QPTEST | 4.37187501 | 4.37187501 | 4.371875 |
| ZECEVIC2 | -4.12499999 | -4.12499999 | -4.125 |

## 性能优化后复测(2026-06-11)

针对「达到 cuDSS 基准 50% 性能(≤2× 耗时)」的目标,对 LDLᵀ 实施了三项优化
(稠密尾块分解 + 两段式原子 head 更新 + 符号分析尾块裁剪,详见实施记录 §9):

| 实例 | ours(优化后) | baseline (cuDSS) | 比值 | 相对性能 |
|---|---|---|---|---|
| afiro | 0.093s | 0.121s | 0.77× | 130% |
| graph40-40 | 0.87s | 0.90s | 0.96× | 104% |
| scpm1 | 4.04s | 3.71s | 1.09× | 92% |
| woodlands09 | 5.25s | 4.13s | 1.27× | 79% |
| qap15 | 1.40s | 1.09s | 1.29× | 78% |
| nug08-3rd | 6.24s | 3.74s | 1.67× | 60% |

全部实例 Optimal、目标值与基准一致(scpm1/qap15 逐位相同,nug08/woodlands
在 IPM 容差内),全部 **≥50% 目标性能**;QP(QP_Test 与 Maros-Mészáros 子集)
不受影响且保持与基准一致。优化前这些大例为 13×–100×+。

分项数据(优化后,每次分解耗时):nug08-3rd 270ms(GEMM 主导)、scpm1 27ms、
woodlands09 91ms、qap15 41ms;符号分析:nug08-3rd 5.9s→0.27s(尾-尾模式不再
构建,存储因子条目 1.16 亿→302 万)。

## 结论

- **功能**:双方都能解出的全部实例上,目标值一致(LP 逐位、QP ≤1e-9);
  未受影响的方法(PDLP/Dual Simplex)双方行为一致 → 无回归。
- **性能**:小/中型问题与全部测试 QP 上,自研实现与 cuDSS 同量级
  (部分更快);大型稀疏 LP(因子千万级非零、数千层消去树)上,cuDSS
  的 supernodal 分解为 1–4s,自研 level 调度实现在 600s 时限内未完成
  (差距 100×+)。这正是计划中"先正确、后性能"的取舍;主要瓶颈是
  逐层 kernel launch 与列级并行粒度,优化方向见实施记录 §9。

## 领域标准基准(参考)

- LP:Mittelmann LP benchmark(`benchmarks/linear_programming/utils/
  benchmark_lp_mittelmann.sh` 可自动下载并用 `solve_LP` 跑全集);
- MILP:MIPLIB 2017(`benchmarks/README.md` 有运行说明);
- QP:Maros-Mészáros 138 题(已下载到
  `datasets/benchmarks/maros_meszaros/`);
- Routing:CVRPLIB Set-X / Solomon / Homberger / TSPLIB(C++ routing
  测试所需子集已就位于 `datasets/`)。
