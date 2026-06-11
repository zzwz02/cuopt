# cuOpt 去除 cuDSS:自研 GPU 稀疏 LDLᵀ 求解器(实施记录)

> 本文档最初是实施计划,现已按实际完成的工作修订为设计与实施记录。
> 实施分支:`feature/remove-cudss-custom-ldlt`(基于 `release/26.06`)。

## 1. 背景与目标

cuOpt 的 Barrier(内点法)求解器(LP / QP / QCQP / SOCP 路径)此前依赖 NVIDIA
cuDSS 闭源库做稀疏对称线性系统的直接分解与求解。目标:**完全移除 cuDSS 依赖**,
用自研 CUDA kernel 实现等价功能,先保证正确性,暂不追求性能。

约束(实施中确认):
1. AMD 重排序用开源实现 —— 最终方案:**链接系统 SuiteSparse 的 `libamd`**
   (apt `libsuitesparse-dev` / conda `suitesparse` / dnf `suitesparse-devel`),
   不内嵌源码、不自研。
2. 保持 CUDA 12.1 工具链(系统 nvcc;A100 驱动 12.9 运行 12.1 产物),代码库中
   与 12.1 不兼容之处一并修复(带版本门控,不影响新工具链)。
3. 尽量保持 cmake 3.30 系列 —— rapids-cmake 26.06 实际要求 ≥3.30.4(而非 4.0),
   仓库内 3 处 `cmake_minimum_required` 降为 3.30,本容器用工作区免安装
   cmake 3.30.8(见 §5)。

## 2. cuDSS 原使用方式(调研结论)

- 唯一封装点 `cpp/src/barrier/sparse_cholesky.cuh`:抽象基类
  `sparse_cholesky_base_t`(analyze/factorize 的 host CSC 与 device CSR 重载、
  solve 的 host/device 重载、`set_positive_definite`)+ cuDSS 实现类。
- 唯一使用者 `cpp/src/barrier/barrier.cu`:恒以
  `set_positive_definite(false)`(LDLᵀ)使用;矩阵为对称 full-view CSR、
  int32 索引、float64 值,两种来源:
  - `!use_augmented`(纯 LP):正规方程 **A·D⁻¹·Aᵀ**(m×m);
  - `use_augmented`(QP/SOCP 或 LP 强制):增广 KKT
    **[[-Q-D-ε_d I, Aᵀ],[A, ε_p I]]**((n+m)×(n+m),正则化后**拟正定**,
    对任意对称排列存在无选主元 LDLᵀ,Vanderbei)。
- 每次 IPM 迭代 1 次 factorize(模式不变)+ 2 次 solve(predictor/corrector);
  外层有 GMRES 迭代精化(`iterative_refinement.hpp`)与自适应正则化兜底。
- 公开参数 `cudss_deterministic`(constants.h / gRPC field 21 / REST 字段)与
  `ordering` 为兼容保留;新实现按构造即确定。

## 3. 替代实现:`sparse_cholesky_ldlt_t`(cpp/src/barrier/sparse_ldlt.cuh)

无数值选主元的稀疏 LDLᵀ:P·A·Pᵀ = L·D·Lᵀ(L 单位下三角不存对角,D 带符号)。

**Host 符号分析**(analyze 一次):
1. 排序:SuiteSparse `amd_order`(`cpp/cmake/thirdparty/FindAMD.cmake` 定位
   `suitesparse/amd.h` + `libamd`,目标 `SuiteSparse::AMD`);失败回退自然排序
   (排列只影响填充量,不影响正确性)。
2. 消去树(Liu,路径压缩)→ 行模式(etree reach,Davis)→ L 的 CSC 模式
   (列内行索引升序)+ CSR 行视图(`rp_ptr/rp_col/rp_pos`,带值位置)。
3. **Level 调度**:`level(j)=1+max(level(children))`;依据 L(j,k)≠0 ⟹ j 是 k
   的 etree 祖先,同层列互相独立。同一层数组同时服务分解与前代/回代。
4. A→因子散射映射 `a2l_`:对称重复条目映射到同一槽位(赋值写,幂等);
   对角条目编码为负数。

**GPU 数值分解**(逐层启动,块↔列):
- 块内对行模式中的 k **顺序**循环(与 host 参考累加次序一致 ⇒ 确定性),
  线程并行处理 L(:,k) 条目,经列内二分查找散射到列 j;层间天然同步。
- 主元规则:NaN/Inf(或要求 SPD 时 ≤0)→ 置失败 flag,factorize 返回 -1
  (语义同 cuDSS info≠0);|d| < τ = 1e-14·max|diag(PAPᵀ)| → **静态选主元**
  (替换为 ±τ 继续,PARDISO/MA57 风格,扰动由上层 GMRES 精化吸收)——
  处理 ADAT 秩亏(冗余/空行)等数学上真奇异的情形。
- 层间检查 `concurrent_halt`(Concurrent 模式可中断)。

**GPU 三角求解**(逐层):排列 gather → 前代(L 的 CSR 行视图,层升序)→
对角缩放 → 回代(CSC 列视图,层降序)→ 排列 scatter;块内固定序树形归约 ⇒
确定性。

**调试后备**:`CUOPT_LDLT_HOST=1` 切换全 host 参考实现(与 GPU 路径同算法
同累加序,良态系统上结果一致到 1e-12,见单测)。

**barrier.cu 配套修改**:
- 实例化新类与显式模板实例化;
- 非 SOC 的增广路径 `dual_perturb` 从 0 改为 1e-8:Q 对角缺失 + 初始障碍项为 0
  时增广 (1,1) 块对角恰为 0,矩阵非拟正定,无选主元 LDLᵀ 必然零主元
  (cuDSS 靠数值选主元掩盖了这一点);自适应正则化的增长改为可从 0 起步。

## 4. 移除与打包

- `sparse_cholesky.cuh` 只留抽象基类;删除 cuDSS 实现、green context、宏。
- CMake:删 `FindCUDSS.cmake` 与全部 cudss 引用;新增 `find_package(AMD REQUIRED)`
  并链接 `SuiteSparse::AMD`(cuopt 与测试目标)。
- 打包:`dependencies.yaml`/conda env/recipe 以 `suitesparse` 替代
  libcudss;wheel 构建脚本 `install_cudss.sh`→`install_suitesparse.sh`,
  auditwheel 不再排除 libcudss(改为自动捆绑 libamd);
  `python/libcuopt/pyproject.toml` 去掉 nvidia-cudss;rpath 去掉 cudss 目录。
- 兼容:`cudss_deterministic` 参数与 gRPC/REST 字段保留(no-op,实现天然确定);
  `ordering` 保留(0/-1/1 现均为 AMD);相关注释改写。

## 5. CUDA 12.1 / gcc 11 / cmake 3.30 兼容修复

工作区 `.toolchain/`(不进版本库)提供:cmake 3.30.8、oneTBB 2021.13、
Boost 1.84(b2 最小安装);bzip2 用 /opt/conda 的。构建命令:

```bash
export PATH=$PWD/.toolchain/cmake-3.30.8-linux-x86_64/bin:$PATH
export CPATH=$PWD/.toolchain/boost-install/include
cmake cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
  -DFETCH_RAPIDS=ON -DBUILD_TESTS=1 -DSKIP_GRPC_BUILD=1 \
  -DBZIP2_ROOT=/opt/conda \
  -DTBB_INCLUDE_DIR=$PWD/.toolchain/oneapi-tbb-2021.13.0/include \
  -DTBB_LIBRARY=$PWD/.toolchain/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8/libtbb.so \
  -DBoost_DIR=$PWD/.toolchain/boost-install/lib/cmake/Boost-1.84.0
cmake --build cpp/build -j48
```

(gRPC 组件因容器无 libgrpc/protobuf 跳过;routing 与 C API 适配器正常构建。)

仓库内的兼容修复(均带编译器/版本门控):
- **CCCL `[[no_unique_address]]`**:nvcc<12.4 cudafe++ ICE("internal error
  during structure layout")→ configure 时对拉取的 CCCL 打补丁。
  ⚠ 补丁必须**对所有编译器一致**禁用:`cuda::mr::any_resource` 等类型跨越
  gcc 编译的 librmm 与 nvcc 编译的 .cu TU 的 ABI 边界,按编译器门控会导致
  同进程两种布局 → 堆损坏(已踩坑并修复)。
- raft `nvtx_range_stack.hpp` 的 NSDMI 被旧 cudafe++ 误解析 → configure 补丁
  改写为显式构造函数。
- 扩展 `__device__` lambda 用于返回类型查询上下文(transform_iterator /
  transform_reduce / inner_product / count_if 及二元归约 op)在 nvcc<12.3 报错
  → `cuda::proclaim_return_type` 包裹(problem.cu、mip_scaling_strategy.cu、
  probing_cache.cu)。
- gcc<13:`omp atomic compare` → `__atomic` CAS 循环(omp_helpers.hpp);
  `omp masked` → `omp master`;`__builtin_cpu_is` 新 CPU 名加版本守卫。
- gcc<12:去掉 `omp task affinity` 提示子句(仅调度提示)。
- barrier/cusparse_view.cu 的 `my_cusparsespmv_preprocess` 调用点补
  `CUDA_VER_12_4_UP` 守卫(定义处本有,调用点漏了;pdlp 同名文件本来就有)。

## 6. 过程中发现并修复的上游潜在 bug

- **`mps_data_model_t::maximize_` 未初始化**(io/mps_data_model.hpp):直接构造
  数据模型且不调 `set_maximize()` 时读未初始化内存(UB);gcc11 下读到非零,
  目标被翻号(cuts.test_cuts_1 把 min -7x1-2x2 当最大化解出 -18 而非 -28)。
  `optimization_problem_t` 等同类成员均有 `{false}`,此处补齐。
- LP-only + tests 构建(`--build-lp-only`)本身链接不过:`branch_and_bound.cpp`
  无条件编入却引用被排除的 `fj_cpu.cu` 符号;改用完整构建绕开,未修
  (上游问题,与本工作无关)。

## 7. 实际提交序列

| Commit | 内容 |
|---|---|
| `docs:` | 实施计划(本文档) |
| `build:` | CUDA 12.1 / GCC 11 / CMake 3.30 工具链兼容修复 |
| `barrier:` | 移除 cuDSS,自研 LDLᵀ(host 符号分析 + host 参考数值)端到端打通;dual_perturb 拟正定修复 |
| `barrier:` | SuiteSparse AMD 填充优化排序(FindAMD.cmake) |
| `barrier:` | GPU 数值分解 kernel(level 调度,确定性) |
| `barrier:` | GPU 三角求解 kernel(确定性) |
| `build:` | 打包/CI 去 cuDSS、引入 suitesparse 依赖 |
| `build:` | CCCL 补丁改为编译器无关(ABI 一致性修复) |
| `io:` | `maximize_` 未初始化修复 |
| `barrier:` | 静态选主元(秩亏系统) |
| `docs:` | 实施记录与验证结果(本次修订) |

## 8. 验证(A100,驱动 12.9)

**C++(系统 CUDA 12.1 工具链构建,完整构建:routing + C API,gRPC 因容器缺
libgrpc 未构建)**

- 全量 `ctest`(无标签过滤)**137 项:136 通过**;唯一失败
  `WAYPOINT_MATRIXTEST` 是上游测试硬编码相对路径 `datasets/...`(ctest 的
  cwd 在构建目录)所致,与本工作无关——在测试 cwd 放置 datasets 符号链接后
  通过,即实际 **137/137**。
- sparse_ldlt 单测 6 个:SPD;拟正定 KKT;device 路径 + 同模式重分解;
  箭头矩阵(非平凡排列);GPU 与 host 参考一致性(良态系统 1e-12);
  奇异矩阵静态选主元(非奇异块解仍准确)。
- C API:`C_API_TEST` 通过。
- gRPC(conda 构建,见下):`GRPC_CLIENT_TEST` 67、`GRPC_INTEGRATION_TEST`
  54、`GRPC_PIPE_SERIALIZATION_TEST` 19 —— 全部通过。

**Python(文档化 conda 环境 all_cuda-129 构建,nvcc 12.9/gcc 14 ——
顺带验证标准工具链下编译干净)**

- `python/cuopt/cuopt/tests`:**108/108 通过**(LP/MILP/QP/SOCP/routing/
  gRPC 远程执行;需 `no_proxy=localhost,127.0.0.1,0.0.0.0`——容器代理会劫持
  localhost gRPC/HTTP 连接,与代码无关)。
- `python/cuopt_server`:**94 通过 + 7 跳过**;7 个跳过全部是上游因
  NVIDIA/cuopt#519 主动禁用的 `test_barrier_solver_options`——临时去掉
  skip 实跑 **7/7 通过**(新实现下该上游问题不复现)。
- `python/cuopt_self_hosted`:**3/3 通过**(需先启动 cuopt_server)。
- `python/libcuopt`:无测试用例。

**与 cuDSS 基准版对比(官方 26.06 wheel,cuDSS 0.7.1)**

详见 `docs/dev/cudss_replacement_benchmark.md`。摘要:双方都解出的实例
目标值一致(LP 逐位、QP ≤1e-9;afiro 双方同为 12 次迭代);QP 12/12 双方
最优且 ours 略快;大型稀疏 LP 上 cuDSS 1–4s vs ours 600s 时限
(性能取舍,见 §9)。Dual Simplex/PDLP 双方行为一致(无意外回归)。

## 9. 性能优化(2026-06-11,达成 cuDSS 50% 性能目标)

初版 level 调度实现与 cuDSS 差 13×–100×+(woodlands09 约 78s/迭代)。
nsys 剖析驱动的三项优化后,LP 基准集全部进入 cuDSS 的 2× 以内
(0.77×–1.67×,见 benchmark 文档):

1. **稠密尾块**:AMD 排序因子的消去树顶部退化为数千列的顺序链
   (woodlands09 3080/qap15 4169/nug08 15052 条链层),链上列又近乎稠密,
   是 level 调度的串行化根源。现将尾部 [j*, n) 作为一整块稠密 LDLᵀ:
   左视分块 panel(每 panel 一次大 k DGEMM,flops 较右视全方阵更新减半)
   + 块内并行的对角块 kernel + shared memory panel 求解;尾宽按
   「头部剩余层数 ≤512」选取,再向下吞并近稠密边界带,显存封顶。
   头列对尾块的 Schur 贡献分轻重两路:尾段短的列成对原子散射,
   尾段长的列稠密化后 DGEMM。求解阶段尾块用 cublasDtrsm。
2. **两段式原子 head 更新**:原"每列一 block、k 顺序"kernel 改为
   update(2D grid:层内列 × 行模式分片,原子累加)+ finalize(选主元
   与缩放)两个 kernel,窄层与长列不再让设备空转。
3. **符号分析尾块裁剪**:先按层数选尾块,尾行的 etree reach 在 j* 截断,
   尾-尾模式完全不构建(nug08-3rd:存储因子条目 1.16 亿→302 万,
   符号分析 5.9s→0.27s)。

原子路径不可逐位复现;`cudss_deterministic` 模式自动回退到原确定性
kernel(无尾块),语义与 cuDSS 一致(确定性换性能)。
`CUOPT_LDLT_TAIL` 可强制/关闭尾宽,`CUOPT_LDLT_STATS` 输出结构与
分阶段计时。

剩余限制:
- 无数值选主元:依赖拟正定性 + 静态选主元 + 调用方自适应正则化与 GMRES 精化;
  极端病态问题可能比 cuDSS 早进入 suboptimal 终止。
- 进一步优化方向:头部 supernodal 化、求解阶段批量 RHS、CUDA Graph 化
  launch 序列。
- gRPC 组件在系统 CUDA 12.1 构建中跳过(容器系统层无 libgrpc/protobuf);
  已在 conda 构建中编译并全部通过(C++ 140 项 + Python 远程执行 11 项)。
