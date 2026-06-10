# cuOpt 去除 cuDSS:自研 GPU 稀疏 LDLᵀ 求解器替代方案

## Context(背景)

cuOpt 的 Barrier(内点法)求解器(LP / QP / QCQP / SOCP 路径)目前依赖 NVIDIA cuDSS 闭源库做稀疏对称线性系统的直接分解与求解。目标:**完全移除 cuDSS 依赖**,用自研 CUDA kernel 实现等价功能(先保证正确性,暂不追求性能)。

用户约束(已确认):
1. **AMD 重排序用开源实现**(CPU 算法)—— 内嵌 SuiteSparse AMD(BSD-3 许可,纯 C,约 10 个文件),不自己重写。
2. **保持当前 CUDA 12.1 工具链**(系统 /usr/local/cuda,nvcc 12.1;GPU A100 驱动 12.9 可运行 12.1 编译产物)。代码库中与 12.1 不兼容之处一并修掉。
3. **尽量保持当前 cmake 3.30.3**;不可行时采用最小侵入的替代方案(见下)。

### 调研结论(已完成)

**cuDSS 的唯一封装点**:`cpp/src/barrier/sparse_cholesky.cuh`
- 抽象基类 `sparse_cholesky_base_t<i_t,f_t>`(L26-36):`analyze()`/`factorize()`(host CSC 与 device CSR 两种重载)、`solve()`(host `dense_vector_t` 与 device `rmm::device_uvector` 两种重载)、`set_positive_definite()`。
- cuDSS 实现 `sparse_cholesky_cudss_t`(L135-897):CSR 全视图(full view)对称矩阵、int32 索引、float64 值;阶段为 REORDERING → SYMBOLIC_FACTORIZATION → FACTORIZATION → SOLVE(单右端项 n×1);分解失败时 `CUDSS_DATA_INFO != 0` 返回 -1;支持 `concurrent_halt` 中断、`cudss_deterministic`、`ordering=1` 时 AMD 重排序。

**唯一使用者**:`cpp/src/barrier/barrier.cu`
- L618 实例化;L620 恒调用 `set_positive_definite(false)` → 实际走 **LDLᵀ**(非 LLᵀ)。
- 被分解的矩阵两种(均为对称、full-view device CSR):
  - `!use_augmented`(纯 LP):正规方程 **A·D⁻¹·Aᵀ**(m×m,准正定);
  - `use_augmented`(QP/SOCP,或 LP 设 augmented=1):增广 KKT **[[-Q-D-εI, Aᵀ],[A, εI]]**((n+m)×(n+m),对称**拟正定/quasi-definite** —— 该类矩阵对任意对称排列均存在无需数值选主元的 LDLᵀ,理论保证来自 Vanderbei)。
- 每次 IPM 迭代:1 次 `factorize`(稀疏模式不变,仅值变化)+ 2 次 `solve`(predictor + corrector)。
- 已有兜底机制(无需新写):分解/求解失败 → barrier.cu L2995-3010 自适应增大正则化扰动重试;`solve` 外层有 GMRES 迭代精化(`iterative_refinement.hpp`,默认开启)清理残差 —— 正好弥补不做数值选主元的精度损失。

**构建/打包接线**(全部要移除):
- `cpp/cmake/thirdparty/FindCUDSS.cmake`(删除)
- `cpp/CMakeLists.txt`:L315 `find_package(CUDSS REQUIRED)`、L516/538/546 include、L564-566 `CUDSS_MT_LIB_FILE_NAME` 编译定义、L600 链接 `${CUDSS_LIB_FILE}`、L737/745 cuopt_cli
- `python/libcuopt/CMakeLists.txt`:rpath `$ORIGIN/../../nvidia/cudss/lib`
- `dependencies.yaml`(libcudss-dev / nvidia-cudss-cu12/cu13)→ 同步 `conda/environments/*.yaml`、`python/libcuopt/pyproject.toml`
- `conda/recipes/libcuopt/recipe.yaml`、`ci/build_wheel_libcuopt.sh`(install_cudss + auditwheel exclude)、`ci/build_wheel_cuopt.sh`、删除 `ci/utils/install_cudss.sh`

**公开 API 兼容性(保留,不破坏)**:
- 参数 `cudss_deterministic`(constants.h `CUOPT_CUDSS_DETERMINISTIC`、gRPC field 21、REST server 字段、`simplex_solver_settings_t::cudss_deterministic`)→ **保留名称与管线**。新实现按确定性设计(level 内列级并行、列内固定顺序累加,不用浮点 atomic),该参数自然满足,变为兼容性 no-op。
- `ordering`(-1 自动 / 0 默认 / 1 AMD)→ 新实现统一用 AMD;参数保留。
- `gf2_presolve.cpp` L37 仅注释提及 cuDSS,顺手改词。

### 工具链现状与对策

| 项 | 现状 | 要求 | 对策 |
|---|---|---|---|
| GPU | A100,驱动 CUDA 12.9 | — | 可运行 12.1 产物(sm_80) |
| nvcc | 12.1(/usr/local/cuda) | 仓库假设 ≥12.9 的分支存在(有 `CMAKE_CUDA_COMPILER_VERSION >= 12.9` 与 `CUDART_VERSION >= 13000` 守卫) | **保持 12.1**;编译报错处加版本守卫修掉(绿色上下文等 CUDA13 代码已有守卫;cuDSS 删除后其 12.9 相关分支多数消失)。CPM 拉取的 raft/rmm/CCCL 26.06 若有 >12.1 的 API 调用,逐处打守卫/替换 |
| cmake | 3.30.3 | 仓库 3 处 `cmake_minimum_required(VERSION 4.0)`(cpp、cmake/RAPIDS.cmake、python/libcuopt);configure 时联网拉 rapids-cmake branch-26.06 及 raft/rmm/CCCL,各自可能再要求 4.0 | **方案 A(先试)**:把仓库内 3 处降为 3.30,尝试 configure;**方案 B(A 失败则用)**:下载官方 cmake≥4.0 免安装 tarball 到工作区 `/home/cuopt-26.06/.toolchain/`(加入 `.git/info/exclude`,不进版本库、不动系统),构建命令显式用该 cmake。结果在 C2 commit message 注明 |
| cuDSS | **系统未安装** | 当前 CMake REQUIRED | 正好:移除后才能构建,无基线构建可比,以测试为准绳 |
| Python 测试 | 无 conda 环境;skill 禁止运行 pip/conda 安装 | — | Python/Server 级测试**本环境不跑**,列为用户环境后续验证项;以 C++ gtest + cuopt_cli 冒烟为主 |

构建迭代命令:`./build.sh libcuopt --build-lp-only --skip-c-python-adapters --skip-grpc-build`(LP-only,跳过 gRPC/适配器,保留测试构建)。测试数据集按 CONTRIBUTING.md 下载并导出 `RAPIDS_DATASET_ROOT_DIR`(允许:仅数据下载,非包安装)。

---

## 设计:自研 `sparse_cholesky_ldlt_t`

新类实现现有 `sparse_cholesky_base_t` 接口,放在新文件 `cpp/src/barrier/sparse_ldlt.cuh`(遵循现有 header-only 模板风格、RMM 分配、handle stream、`_t` 命名)。`sparse_cholesky.cuh` 保留基类、删除 cuDSS 实现与 `#include "cudss.h"`/宏。

### 算法结构(正确性优先)

**Host 符号分析**(CPU,analyze 时一次,性能无关紧要):
1. device CSR(full view)→ host;按排列后下三角提取模式。
2. **重排序:SuiteSparse AMD**(内嵌开源 C 代码,`amd_order()`)。性质:排列只影响填充量(性能),**不影响正确性**,任何合法排列结果都正确。
3. 消去树(etree)+ 后序 + 列计数 + L 的符号结构(CSC)—— Davis《Direct Methods for Sparse Linear Systems》标准算法,~350 行,自写(纯 CPU,无许可问题)。
4. **Level scheduling**:`level(j) = 1 + max(level(etree-children))`;依据定理 L(j,k)≠0 ⟹ j 是 k 的祖先,同 level 列互相独立可并行。同一 level 数组同时服务前代/回代三角求解(回代按 level 逆序)。
5. 预计算散射映射 `A_entry → L_slot`:对输入 CSR 每个条目 e=(i,j),排列后落在下三角的那一侧(pi≥pj)映射到 L 存储位;对称重复侧置 -1。factorize 时一个 kernel 刷新数值。
6. 上传 device:L 模式(CSC + CSR 两份,分别服务数值分解/前代与回代)、level 数组、散射映射、排列向量。

**Device 数值分解**(自研 kernel,逐 level 启动):
- Left-looking 列式 LDLᵀ:每 level 一次 kernel,block ↔ 列 j;block 内对 j 行模式中的 k **顺序**循环(保证确定性),线程并行处理 L(:,k) 条目,目标位置经列内行索引二分查找;完成后 d_j = c(j),`L(i,j) = c(i)/d_j`。
- `|d_j|` 过小或非有限 → 全局 flag 置位,`factorize` 返回 -1(与 cuDSS info≠0 语义一致,触发 barrier 已有的正则化加重重试)。
- 每个 level 间检查 `settings_.concurrent_halt`,返回 `CONCURRENT_HALT_RETURN`(Concurrent 模式必需)。

**Device 三角求解**(自研 kernel):
- 排列 b → 前代(L,CSR pull 式、level 顺序、块内固定序归约)→ 对角缩放(÷d)→ 回代(Lᵀ = L 的 CSC,level 逆序)→ 逆排列。全程确定性。

**CPU 参考实现**(保留作调试/校验后备):同类内实现 host 完整 LDLᵀ(简单 up-looking,~150 行),由环境变量切换;开发期先用它打通端到端,再换 GPU kernel,两者互为黄金参考。

**Host CSC 接口重载**:转换为 device CSR 后复用 device 路径(与 cuDSS 类原行为一致)。

### 复用的现有设施
- `iterative_refinement.hpp` GMRES 精化(调用方已接好,无需改)
- `device_sparse_matrix.cuh` 的 `device_csr_matrix_t`(`row_start/j/x`)、`csc_matrix_t::to_compressed_row()`、`dense_vector_t`、`dense_matrix.hpp::chol`(单测稠密参考)
- barrier.cu 自适应扰动重试(L2995-3010)、`CONCURRENT_HALT_RETURN`
- `simplex_solver_settings_t` 的 `ordering` / `concurrent_halt` / `cudss_deterministic` 字段

### 内嵌 SuiteSparse AMD
- 来源:github.com/DrTimothyAldenDavis/SuiteSparse 的 `AMD/`(BSD-3-clause)+ 所需的最小 `SuiteSparse_config` 子集。
- 放置:`cpp/src/barrier/suitesparse_amd/`(保留原版权头 + 目录级 LICENSE 说明),以 C 源文件编入 cuopt 目标。
- 接口:`amd_order(n, Ap, Ai, P, ...)`,输入对称模式(上/下三角均可),输出排列 P。

---

## 实施步骤(每步一个 git commit)

> 新建分支 `feature/remove-cudss-custom-ldlt`(基于当前 `release/26.06`);所有 commit 带 DCO `-s` 与 Claude co-author 行;**只 commit 不 push**。
> 步骤 0(无 commit):cmake 方案 A/B 落定;下载测试数据集、导出 `RAPIDS_DATASET_ROOT_DIR`。

| # | Commit 内容 | 验证门槛 |
|---|---|---|
| C1 | 把本计划放入仓库 `docs/dev/cudss_replacement_plan.md` | — |
| C2 | **移除 cuDSS + CPU 参考实现打通端到端 + 工具链兼容修复**:新增 `sparse_ldlt.cuh`(类骨架 + host 符号分析:etree/列计数/符号 L/level + 暂用自然排序 + CPU 数值 LDLᵀ/三角求解);`sparse_cholesky.cuh` 删 cuDSS 留基类;`barrier.cu` L618 换实例化、L4681 模板实例化更新;CMake/FindCUDSS/rpath 去除 cuDSS;**CUDA 12.1 / cmake 兼容性修复**(若方案 A 成立含 3 处版本降级);新增 gtest `cpp/tests/dual_simplex/unit_tests/sparse_ldlt_test.cu`(小型 SPD/拟正定矩阵 vs 稠密参考,残差 < 1e-8) | `./build.sh libcuopt --build-lp-only ...` 构建成功;新 gtest 过;`ctest -R solve_barrier` 过 |
| C3 | **内嵌 SuiteSparse AMD** 并接入符号分析(替换自然排序)+ 单测(排列合法性、填充量 ≤ 自然排序) | gtest 过;barrier 测试仍过 |
| C4 | **GPU 数值分解 kernel**(level-scheduled left-looking LDLᵀ、确定性、halt 检查、失败 flag)+ 与 CPU 参考对比单测 | gtest(GPU vs CPU 一致)过;solve_barrier 过 |
| C5 | **GPU 三角求解 kernel**(排列/前代/对角/回代/逆排列);默认全 GPU 路径,CPU 路径留调试开关 | gtest 过;`ctest -R "solve_barrier\|socp"` 过 |
| C6 | **打包/CI/依赖清理**:`dependencies.yaml`、`conda/recipes/libcuopt/recipe.yaml`、同步 `conda/environments/*.yaml` 与 `python/libcuopt/pyproject.toml`(生成器不可用则一致性手改并注明)、`ci/build_wheel_*.sh`、删 `ci/utils/install_cudss.sh`、`gf2_presolve.cpp` 注释 | 全仓 grep 无残留 cudss(除历史记录与兼容参数名) |
| C7 | **全量验证修补**:跑全部相关 C++ 测试与 CLI 冒烟并修复问题 | 见下方验证清单 |

## 验证(C7 验收清单)

1. C++:`ctest --test-dir cpp/build -R "solve_barrier"`、SOCP 测试(`cpp/tests/socp/solve_barrier_socp.cu`)、新 sparse_ldlt 单测;LP 数据集回归(afiro 等,barrier 与 PDLP/DualSimplex 目标值交叉比对,相对容差 1e-4)。
2. CLI 冒烟:`cuopt_cli <mps> --method 3`(Barrier)与 method 1/2 目标一致;QP MPS 一例。
3. Concurrent 方法冒烟(method=0):barrier 与 PDLP 并发、halt 中断不挂死。
4. 全仓搜索确认无残留 `cudss`(除 RELEASE-NOTES.md 历史记录与兼容参数名 `cudss_deterministic`)。
5. (本环境不可行,移交用户环境)`pytest test_qp.py / test_socp.py / test_lp.py::test_barrier_solver_options`。

## 风险与对策

- **无数值选主元的 LDLᵀ 遇病态矩阵**:拟正定性 + 现有自适应正则化 + GMRES 精化兜底;失败语义与 cuDSS 对齐(-1 触发重试);必要时类内加对角 |d| 下限静态正则。
- **CUDA 12.1 与 raft/rmm/CCCL 26.06 的 API 差异**:编译期暴露,逐处守卫/替换;若出现深层不兼容(如 CCCL 硬性要求新版 nvcc),如实报告并给出最小可行替代(如锁旧版 CCCL)。
- **cmake 方案 A 失败**:fetched rapids-cmake/raft/rmm 各自要求 4.0 的可能性高;则走方案 B(工作区本地 cmake tarball,不动系统、不进版本库)。
- **AMD(SuiteSparse)集成**:成熟开源代码,风险低;只损性能不损正确性,可随时回退自然排序。
- **Level 串行化极端情形**(etree 退化为链):仅慢不错;本阶段接受。
- **数据集下载失败**:验证降级为"编译通过 + 新单测 + 不依赖数据集的 barrier 单测",如实说明。

---

# 实施结果(附录,2026-06-10)

## 最终状态

cuDSS 已完全移除,barrier 求解器改用自研 `sparse_cholesky_ldlt_t`
(`cpp/src/barrier/sparse_ldlt.cuh`):

- **符号分析(host)**:SuiteSparse AMD 排序(系统 `libamd`,apt
  `libsuitesparse-dev` / conda `suitesparse`)+ 消去树 + 行模式 + level 调度
  + A→L 散射映射。
- **数值分解(GPU)**:level 调度的 left-looking LDLᵀ kernel;块内 k 循环顺序
  执行,确定性;NaN/Inf 主元报错,近零主元做静态选主元
  (τ = 1e-14·max|diag|,PARDISO/MA57 风格),扰动由调用方 GMRES 精化吸收。
- **三角求解(GPU)**:排列 gather → 前代(L 的 CSR 行视图,level 升序)→
  对角缩放 → 回代(CSC 列视图,level 降序)→ 排列 scatter;固定序块内归约,
  确定性。
- `CUOPT_LDLT_HOST=1` 环境变量切换到全 host 参考实现(调试用)。
- 公开参数 `cudss_deterministic`、`ordering` 等保留(兼容),新实现天然确定。

与计划的偏差:
- AMD 改为链接系统 SuiteSparse(用户要求),不内嵌源码;新增
  `cpp/cmake/thirdparty/FindAMD.cmake`。
- 修复了两个被新工具链暴露的上游潜在 bug:
  `mps_data_model_t::maximize_` 未初始化(UB);
  LP-only+tests 构建的 fj_cpu 符号缺失改用完整构建绕开(未修)。

## 本容器的构建环境(CUDA 12.1 / gcc 11.4)

仓库不可用系统 cmake 3.30.3(rapids-cmake 26.06 要求 ≥3.30.4),工作区
`.toolchain/` 放置了免安装工具(不进版本库):cmake 3.30.8、oneTBB 2021.13、
Boost 1.84(b2 最小安装:headers + iostreams/program_options/serialization)。

构建命令:

```bash
export PATH=$PWD/.toolchain/cmake-3.30.8-linux-x86_64/bin:$PATH
export CPATH=$PWD/.toolchain/boost-install/include
cmake cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
  -DFETCH_RAPIDS=ON -DBUILD_TESTS=1 \
  -DSKIP_ROUTING_BUILD=1 -DSKIP_GRPC_BUILD=1 -DSKIP_C_PYTHON_ADAPTERS=1 \
  -DBZIP2_ROOT=/opt/conda \
  -DTBB_INCLUDE_DIR=$PWD/.toolchain/oneapi-tbb-2021.13.0/include \
  -DTBB_LIBRARY=$PWD/.toolchain/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8/libtbb.so \
  -DBoost_DIR=$PWD/.toolchain/boost-install/lib/cmake/Boost-1.84.0
cmake --build cpp/build -j48
```

CUDA 12.1 兼容性修复(均已提交,自动按版本门控):
- CCCL `[[no_unique_address]]` 在 nvcc<12.4 触发 ICE → configure 时对拉取的
  CCCL 打补丁,**对所有编译器一致禁用**(librmm 与 .cu TU 的 ABI 必须一致);
- raft `nvtx_range_stack.hpp` NSDMI 被旧 cudafe++ 误解析 → 改写为显式构造;
- 扩展 device lambda 的返回类型查询(nvcc<12.3)→ `cuda::proclaim_return_type`;
- gcc<13:`omp atomic compare`→`__atomic` CAS、`omp masked`→`master`、
  新 CPU 名守卫;gcc<12:去掉 `omp task affinity` 提示子句。

## 验证结果(A100,CUDA 12.1)

- `ctest -L numopt` 22 个测试套件全部通过(串行;含 DUAL_SIMPLEX/BARRIER、
  QP、SOCP、PDLP、LP、MIP、CUTS、确定性测试等)。
- sparse_ldlt 单测 6 个:SPD、拟正定 KKT、device 路径+同模式重分解、
  箭头矩阵(非平凡排列)、GPU=host 参考一致性(良态系统 1e-12)、
  奇异矩阵静态选主元。
- CLI 冒烟:afiro 上 method 0/1/2/3 目标一致(barrier -464.753135 vs
  dual simplex -464.753143);QP_Test_1.qps 最优 -99.96,残差 1e-10 量级;
  Concurrent 模式 barrier 正常参与、halt 不挂死。
- woodlands09(11.4 万约束,因子 1.16e7 非零,3401 层消去树):IPM 残差
  单调收敛(primal 2e+01→2e-04 / 10 迭代),静态选主元生效。

## 已知限制

- **性能**:level 调度逐层启动 kernel,每次分解/求解需 O(层数) 次 launch;
  woodlands09 约 78s/迭代(cuDSS 亚秒级)。计划内取舍("先不考虑性能");
  后续可做:层内多列融合、launch graph 化、supernodal 化、块状求解。
- 无数值选主元:依赖拟正定性 + 静态选主元 + 调用方自适应正则化与 GMRES 精化。
- Python / server 级测试未在本容器运行(无 conda 环境;按 skill 政策不安装
  包),需在标准开发环境补跑:
  `pytest python/cuopt/cuopt/tests/{quadratic_programming,socp}`、
  `python/cuopt_server/.../test_lp.py::test_barrier_solver_options`。
