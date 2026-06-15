# QP 全集性能优化报告（vs 官方 cuDSS 版 cuOpt）

> 关联文档:cuDSS 替换实施记录见 [cudss_replacement_plan.md](cudss_replacement_plan.md),LP+QP 基准快照见 [cudss_replacement_benchmark.md](cudss_replacement_benchmark.md)。

> 分支 `feature/remove-cudss-custom-ldlt`。自研 LDLᵀ（`cpp/src/barrier/sparse_ldlt.cuh`）
> 替代 cuDSS 后的 Maros-Mészáros 全集（138 题）逐题性能对账。
> 协议：Barrier（`--method 3`），`--time-limit 180`，A100-PCIE-40GB（锁频 1410MHz）。
> 对照基线：官方 26.06 wheel（`libcuopt-cu12==26.6.0`，cuDSS 0.7.1），`.toolchain/baseline-venv`。
> 测量脚本：`/tmp/bench_qp_status.sh <cli> <outdir>`。本版列 = `cpp/build/cuopt_cli`。

## 当前汇总（v8，本轮：解析正确性 + 调度细化）

| 指标 | 值 |
|---|---|
| 可比题数（双方均有耗时） | 136 |
| **本版 ≥100%（≤cuDSS 耗时）** | **100 / 136** |
| 本版慢于 cuDSS | 36 / 136 |
| 双方完成总耗时 | 本版 103.2s vs cuDSS 49.1s |
| 单题耗时比中位数（本版/cuDSS） | **0.778**（中位题本版更快） |
| 目标值不一致（双方 Optimal，rel>1e-4） | **0** |
| 正确性回归（本版失败而 cuDSS 成功） | **0** |

### 正确性结论：本版 ≥ cuDSS

- **零回归**：无「本版未达 Optimal 而 cuDSS 达 Optimal」的题。
- **零错解**：130 道双方 Optimal 题目标值全部 rel ≤ 1e-4 一致。
- **三处优于 cuDSS**：
  - `DPKLO1` / `QFORPLAN`：cuDSS 官方版 ParseError，本版正确解析并求得 Optimal
    （RHS 自由格式行 token 奇偶消歧 + MPS 固定格式自动回退，`cpp/src/io/`）。
  - `STADAT1`：cuDSS Suboptimal，本版 Optimal。
- 本版所有未达 Optimal 的题（`CVXQP1_L` Suboptimal、`PRIMAL1-4` NumError、`UBH1` Suboptimal）
  与 cuDSS **同状态**，属两侧共有的病态题，非本版缺陷。

## 优化重点（按绝对耗时差 / 倍数差）

绝对差最大：`CVXQP1_L`(+32.7s)、`CVXQP3_L`(+11.7s)、`BOYD2`(+4.2s)、`CONT-300`(+2.6s)、
`CVXQP2_L`(+1.6s)、`CONT-201/200`(+1.3/1.1s)、`QGROW22`(+0.9s)。
倍数差最大：`CVXQP1_L`(6.4×)、`CONT-300`(4.3×)、`QGROW22`(4.1×)、`CVXQP3_L`(4.0×)、
`CONT-201`(3.6×)、`CONT-200`(3.4×)。

题型聚类：① CVXQP_L 系（稠密 KKT、GMRES 内迭代多）；② CONT 系（宽网格 PDE 型）；
③ BOYD2（n≈466k 巨型）；④ QGROW 系；⑤ 小题固定开销尾部（比值 1.0–1.9，绝对差 ≤0.1s）。

## 已落地的优化（累计，自 cuDSS 替换基线起）

1. **结构化枢轴规则**（quasi-definite 符号下限 / SPD 仅丢弃微小正枢轴、保留消去型负枢轴 / 旧式
   兼容初始点系统）：修复 ~15 题 IPM 迭代爆炸（314→20 级），目标值与 cuDSS 一致。
2. **对称 Jacobi 缩放**（M'=SMS，仅 quasi-definite/旧式路径）：抑制无主元 LDLᵀ 在病态系统上的
   元素增长（QSHELL 4.7→1.2s）。
3. **精确列计数（Gilbert-Ng-Peyton）驱动的密集尾块**（边际密度窗口判据）：宽网格不再把近空区域
   当稠密块分解（LISWET 9487²→0；CONT-300 尾 3117→1442）。
4. **层捆绑 + 链窗口调度**：数千个 per-level kernel 启动合为单块内 `__syncthreads` 屏障执行
   （链型/带状 etree）；factor 捆绑按 scatter 二分搜索深度计权，长列不入捆绑。
5. **带状嵌套剖分排序**（自然或 RCM 序下带宽 ≤40 时，递归二分 + 带宽分隔符替代 AMD）：
   etree 深度 O(n)→O(bw·log n)（LISWET 9999→54 层，7.0→0.42s）。
6. **CUDA Graph 捕获**因式分解与三角求解 kernel 序列（缩放路径静态），消除逐次 launch 延迟。
7. **极端初始点鲁棒化**：非有限 barrier 对角线钳制、初始解幅值封顶（QGROW15 NumError→Optimal）。
8. **解析器修复**：RHS 自由格式消歧 + 固定格式回退（DPKLO1/QFORPLAN）。

## 逐题对照（全 138 题）

| 题 | 本版(s) | cuDSS(s) | 比值 | 本版状态 | cuDSS状态 |
|---|---|---|---|---|---|
| AUG2D | 0.20 | 0.20 | 1.00 | Optimal | Optimal |
| AUG2DC | 0.20 | 0.20 | 1.00 | Optimal | Optimal |
| AUG2DCQP | 0.28 | 0.23 | 1.22 | Optimal | Optimal |
| AUG2DQP | 0.28 | 0.25 | 1.12 | Optimal | Optimal |
| AUG3D | 0.14 | 0.19 | 0.74 | Optimal | Optimal |
| AUG3DC | 0.14 | 0.16 | 0.88 | Optimal | Optimal |
| AUG3DCQP | 0.16 | 0.18 | 0.89 | Optimal | Optimal |
| AUG3DQP | 0.16 | 0.22 | 0.73 | Optimal | Optimal |
| BOYD1 | 0.24 | 0.28 | 0.86 | Optimal | Optimal |
| BOYD2 | 6.93 | 2.76 | 2.51 | Optimal | Optimal |
| CONT-050 | 0.17 | 0.22 | 0.77 | Optimal | Optimal |
| CONT-100 | 0.43 | 0.26 | 1.65 | Optimal | Optimal |
| CONT-101 | 0.43 | 0.32 | 1.34 | Optimal | Optimal |
| CONT-200 | 1.55 | 0.46 | 3.37 | Optimal | Optimal |
| CONT-201 | 1.76 | 0.49 | 3.59 | Optimal | Optimal |
| CONT-300 | 3.43 | 0.80 | 4.29 | Optimal | Optimal |
| CVXQP1_L | 38.74 | 6.04 | 6.41 | Suboptimal | Suboptimal |
| CVXQP1_M | 0.36 | 0.54 | 0.67 | Optimal | Optimal |
| CVXQP1_S | 0.09 | 0.19 | 0.47 | Optimal | Optimal |
| CVXQP2_L | 2.46 | 0.90 | 2.73 | Optimal | Optimal |
| CVXQP2_M | 0.16 | 0.30 | 0.53 | Optimal | Optimal |
| CVXQP2_S | 0.09 | 0.12 | 0.75 | Optimal | Optimal |
| CVXQP3_L | 15.57 | 3.87 | 4.02 | Optimal | Optimal |
| CVXQP3_M | 0.64 | 0.87 | 0.74 | Optimal | Optimal |
| CVXQP3_S | 0.12 | 0.17 | 0.71 | Optimal | Optimal |
| DPKLO1 | 0.09 | — | — | Optimal | ParseError |
| DTOC3 | 0.12 | 0.21 | 0.57 | Optimal | Optimal |
| DUAL1 | 0.11 | 0.11 | 1.00 | Optimal | Optimal |
| DUAL2 | 0.10 | 0.11 | 0.91 | Optimal | Optimal |
| DUAL3 | 0.11 | 0.11 | 1.00 | Optimal | Optimal |
| DUAL4 | 0.10 | 0.11 | 0.91 | Optimal | Optimal |
| DUALC1 | 0.10 | 0.18 | 0.56 | Optimal | Optimal |
| DUALC2 | 0.15 | 0.19 | 0.79 | Optimal | Optimal |
| DUALC5 | 0.08 | 0.11 | 0.73 | Optimal | Optimal |
| DUALC8 | 0.09 | 0.17 | 0.53 | Optimal | Optimal |
| EXDATA | 0.49 | 0.68 | 0.72 | Optimal | Optimal |
| GENHS28 | 0.05 | 0.08 | 0.62 | Optimal | Optimal |
| GOULDQP2 | 0.08 | 0.21 | 0.38 | Optimal | Optimal |
| GOULDQP3 | 0.11 | 0.23 | 0.48 | Optimal | Optimal |
| HS118 | 0.07 | 0.11 | 0.64 | Optimal | Optimal |
| HS21 | 0.07 | 0.13 | 0.54 | Optimal | Optimal |
| HS268 | 0.06 | 0.10 | 0.60 | Optimal | Optimal |
| HS35 | 0.05 | 0.09 | 0.56 | Optimal | Optimal |
| HS35MOD | 0.06 | 0.10 | 0.60 | Optimal | Optimal |
| HS51 | 0.04 | 0.09 | 0.44 | Optimal | Optimal |
| HS52 | 0.04 | 0.08 | 0.50 | Optimal | Optimal |
| HS53 | 0.06 | 0.09 | 0.67 | Optimal | Optimal |
| HS76 | 0.05 | 0.09 | 0.56 | Optimal | Optimal |
| HUES-MOD | 0.08 | 0.12 | 0.67 | Optimal | Optimal |
| HUESTIS | 0.09 | 0.12 | 0.75 | Optimal | Optimal |
| KSIP | 0.28 | 0.26 | 1.08 | Optimal | Optimal |
| LASER | 0.18 | 0.29 | 0.62 | Optimal | Optimal |
| LISWET1 | 0.29 | 0.39 | 0.74 | Optimal | Optimal |
| LISWET10 | 0.31 | 0.40 | 0.77 | Optimal | Optimal |
| LISWET11 | 0.27 | 0.38 | 0.71 | Optimal | Optimal |
| LISWET12 | 0.39 | 0.47 | 0.83 | Optimal | Optimal |
| LISWET2 | 0.12 | 0.26 | 0.46 | Optimal | Optimal |
| LISWET3 | 0.12 | 0.23 | 0.52 | Optimal | Optimal |
| LISWET4 | 0.12 | 0.22 | 0.55 | Optimal | Optimal |
| LISWET5 | 0.12 | 0.25 | 0.48 | Optimal | Optimal |
| LISWET6 | 0.12 | 0.22 | 0.55 | Optimal | Optimal |
| LISWET7 | 0.27 | 0.32 | 0.84 | Optimal | Optimal |
| LISWET8 | 0.36 | 0.40 | 0.90 | Optimal | Optimal |
| LISWET9 | 0.37 | 0.37 | 1.00 | Optimal | Optimal |
| LOTSCHD | 0.06 | 0.10 | 0.60 | Optimal | Optimal |
| MOSARQP1 | 0.23 | 0.20 | 1.15 | Optimal | Optimal |
| MOSARQP2 | 0.12 | 0.16 | 0.75 | Optimal | Optimal |
| POWELL20 | 0.28 | 0.36 | 0.78 | Optimal | Optimal |
| PRIMAL1 | 0.11 | 0.13 | 0.85 | NumError | NumError |
| PRIMAL2 | 0.24 | 0.21 | 1.14 | NumError | NumError |
| PRIMAL3 | 0.09 | 0.12 | 0.75 | NumError | NumError |
| PRIMAL4 | 0.08 | 0.11 | 0.73 | NumError | NumError |
| PRIMALC1 | 0.10 | 0.14 | 0.71 | Optimal | Optimal |
| PRIMALC2 | 0.10 | 0.14 | 0.71 | Optimal | Optimal |
| PRIMALC5 | 0.09 | 0.12 | 0.75 | Optimal | Optimal |
| PRIMALC8 | 0.11 | 0.14 | 0.79 | Optimal | Optimal |
| Q25FV47 | 1.47 | 0.86 | 1.71 | Optimal | Optimal |
| QADLITTL | 0.10 | 0.18 | 0.56 | Optimal | Optimal |
| QAFIRO | 0.06 | 0.13 | 0.46 | Optimal | Optimal |
| QBANDM | 0.17 | 0.18 | 0.94 | Optimal | Optimal |
| QBEACONF | 0.15 | 0.18 | 0.83 | Optimal | Optimal |
| QBORE3D | 0.18 | 0.23 | 0.78 | Optimal | Optimal |
| QBRANDY | 0.15 | 0.17 | 0.88 | Optimal | Optimal |
| QCAPRI | 0.72 | 0.65 | 1.11 | Optimal | Optimal |
| QE226 | 0.18 | 0.20 | 0.90 | Optimal | Optimal |
| QETAMACR | 0.47 | 0.62 | 0.76 | Optimal | Optimal |
| QFFFFF80 | 0.37 | 0.54 | 0.69 | Optimal | Optimal |
| QFORPLAN | 0.80 | — | — | Optimal | ParseError |
| QGFRDXPN | 0.61 | 0.57 | 1.07 | Optimal | Optimal |
| QGROW15 | 0.93 | 0.35 | 2.66 | Optimal | Optimal |
| QGROW22 | 1.18 | 0.29 | 4.07 | Optimal | Optimal |
| QGROW7 | 0.55 | 0.25 | 2.20 | Optimal | Optimal |
| QISRAEL | 0.18 | 0.22 | 0.82 | Optimal | Optimal |
| QPCBLEND | 0.10 | 0.12 | 0.83 | Optimal | Optimal |
| QPCBOEI1 | 0.22 | 0.18 | 1.22 | Optimal | Optimal |
| QPCBOEI2 | 0.17 | 0.17 | 1.00 | Optimal | Optimal |
| QPCSTAIR | 0.17 | 0.18 | 0.94 | Optimal | Optimal |
| QPILOTNO | 1.15 | 1.13 | 1.02 | Optimal | Optimal |
| QPTEST | 0.05 | 0.09 | 0.56 | Optimal | Optimal |
| QRECIPE | 0.13 | 0.22 | 0.59 | Optimal | Optimal |
| QSC205 | 0.10 | 0.17 | 0.59 | Optimal | Optimal |
| QSCAGR25 | 0.21 | 0.18 | 1.17 | Optimal | Optimal |
| QSCAGR7 | 0.11 | 0.20 | 0.55 | Optimal | Optimal |
| QSCFXM1 | 0.44 | 0.34 | 1.29 | Optimal | Optimal |
| QSCFXM2 | 0.73 | 0.40 | 1.82 | Optimal | Optimal |
| QSCFXM3 | 0.89 | 0.51 | 1.75 | Optimal | Optimal |
| QSCORPIO | 0.18 | 0.24 | 0.75 | Optimal | Optimal |
| QSCRS8 | 0.28 | 0.30 | 0.93 | Optimal | Optimal |
| QSCSD1 | 0.12 | 0.16 | 0.75 | Optimal | Optimal |
| QSCSD6 | 0.12 | 0.19 | 0.63 | Optimal | Optimal |
| QSCSD8 | 0.31 | 0.21 | 1.48 | Optimal | Optimal |
| QSCTAP1 | 0.15 | 0.24 | 0.62 | Optimal | Optimal |
| QSCTAP2 | 0.12 | 0.22 | 0.55 | Optimal | Optimal |
| QSCTAP3 | 0.12 | 0.23 | 0.52 | Optimal | Optimal |
| QSEBA | 0.28 | 0.33 | 0.85 | Optimal | Optimal |
| QSHARE1B | 0.13 | 0.21 | 0.62 | Optimal | Optimal |
| QSHARE2B | 0.14 | 0.22 | 0.64 | Optimal | Optimal |
| QSHELL | 1.02 | 1.22 | 0.84 | Optimal | Optimal |
| QSHIP04L | 0.16 | 0.21 | 0.76 | Optimal | Optimal |
| QSHIP04S | 0.16 | 0.22 | 0.73 | Optimal | Optimal |
| QSHIP08L | 0.61 | 0.46 | 1.33 | Optimal | Optimal |
| QSHIP08S | 0.37 | 0.32 | 1.16 | Optimal | Optimal |
| QSHIP12L | 0.77 | 0.40 | 1.93 | Optimal | Optimal |
| QSHIP12S | 0.32 | 0.29 | 1.10 | Optimal | Optimal |
| QSIERRA | 0.59 | 0.53 | 1.11 | Optimal | Optimal |
| QSTAIR | 0.28 | 0.36 | 0.78 | Optimal | Optimal |
| QSTANDAT | 0.34 | 0.33 | 1.03 | Optimal | Optimal |
| S268 | 0.06 | 0.10 | 0.60 | Optimal | Optimal |
| STADAT1 | 0.50 | 0.36 | 1.39 | Optimal | Suboptimal |
| STADAT2 | 0.20 | 0.23 | 0.87 | Optimal | Optimal |
| STADAT3 | 0.19 | 0.22 | 0.86 | Optimal | Optimal |
| STCQP1 | 0.42 | 0.31 | 1.35 | Optimal | Optimal |
| STCQP2 | 0.57 | 0.35 | 1.63 | Optimal | Optimal |
| TAME | 0.04 | 0.08 | 0.50 | Optimal | Optimal |
| UBH1 | 0.17 | 0.22 | 0.77 | Suboptimal | Suboptimal |
| VALUES | 0.10 | 0.14 | 0.71 | Optimal | Optimal |
| YAO | 0.32 | 0.32 | 1.00 | Optimal | Optimal |
| ZECEVIC2 | 0.07 | 0.10 | 0.70 | Optimal | Optimal |

