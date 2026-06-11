# cuOpt 去除 RMM / RAFT / RAPIDS 依赖：可行性评估与实施计划

> 本文档为可行性评估与实施计划（尚未实施）。评估方法：7 个维度并行调研
> （RMM、RAFT core、RAFT primitives、构建基础设施、Python 层、替换架构设计、
> 仓库上下文/cuDSS 先例）+ 综合 + 交叉审查，所有承重论断经过仓库实证抽查。
> 基线分支：`feature/remove-cudss-custom-ldlt`（cuDSS 已移除，见
> `docs/dev/cudss_replacement_plan.md`）。评估日期：2026-06。

## 1. 结论

**可行，且风险可控。** 与 cuDSS 移除不同，这次不需要发明任何新算法——cuOpt
对 RAPIDS 的使用是「宽而浅」的：约 6,650 个 `rmm::`/`raft::` 调用点中超过 75%
集中在 5 个符号上（`device_uvector`、`device_span`、`raft::copy`、`handle_t`、
`nvtx::range`）。本质上这是一个 **shim + 脚本化改名问题，不是算法问题**。

工作量估算：

- 核心 C++/CMake/Cython 路径：**12–16 人周**；其中构建/打包若全部做完
  （vendored logger、wheel 命名、export config——这些项无法被改名技巧压缩）
  需再加 ~3–4 周；
- 全量移除（含 routing Python API 的 cudf 迁移）：**16–24 人周**；
- CI 与 rapidsai/shared-workflows 脱钩：另计 **+2–3 周**，可推迟，不影响交付的
  二进制。

## 2. 依赖现状盘点（实测，已排除 build 目录污染）

| 指标 | 数值 |
|---|---|
| `rmm::` 出现次数 / 文件数 | ~3,000 / ~230 文件，22 个符号 |
| `raft::` 出现次数 / 文件数 | ~3,650 / ~315 文件，75 个符号 |
| `RAFT_*` 错误宏调用点 | 772（`RAFT_CUDA_TRY` 362、`RAFT_CHECK_CUDA` 215、`RAFT_CUSPARSE_TRY` 133、`RAFT_CUBLAS_TRY` 57；`RAFT_EXPECTS`/`RAFT_FAIL` 为 0——cuOpt 已有自己的 `cuopt_expects`） |
| 合计改名面 | ~6,650 个调用点，涉及约 370–550 个文件（两种去重口径） |
| 头部符号 | `rmm::device_uvector` 2,207、`raft::device_span` 1,328、`raft::copy` 638、`raft::handle_t` ~616、`nvtx::range` 449、`cuda_stream_view` 310、`device_scalar` 204 |
| 最重模块 | routing（~109/158 文件）、mip_heuristics（~83/115）、pdlp（~44/48）、barrier；**dual_simplex（10/47）与 branch_and_bound（3/15）几乎干净**；MPS parser 零实际使用 |
| 公共 C++ 头暴露 | `cpp/include/cuopt` 下 **22 个头文件** 含 rmm/raft 类型 |
| C API | **`cuopt_c.h` 零暴露**（handle 在 `cuopt_c_internal.hpp:34` 内部构造，已验证） |
| Cython/Python | 8 个文件 cimport pylibraft handle、41 个 `DeviceBuffer.c_from_unique_ptr` 所有权转移点；routing 公共 API 以 cudf 为输入/输出类型 |
| CI/打包 | ~145 个 rapids-\* gha-tools 调用（18 种）、`.github/workflows` 中 32 处 rapidsai/shared-workflows 引用、rapids-build-backend 驱动 3 个 wheel、dependencies.yaml 818 行 |
| 现成逃生口 | **没有**。`FETCH_RAPIDS=OFF` 只是改为 `find_package(CCCL/RMM/RAFT REQUIRED)`（cpp/CMakeLists.txt:244-258），仍依赖 RAPIDS；源代码中零 `#ifdef` 隔离 |

## 3. 关键事实

1. **RAFT 是纯编译期 header-only 依赖；RMM 不是。** `get_raft.cmake` 以
   `RAFT_COMPILE_LIBRARY OFF` 构建，ldd 中没有 libraft.so；但 ldd 实测证明
   **librmm.so 和 librapids_logger.so 是 libcuopt.so 的硬 DT_NEEDED 运行时
   依赖**。移除后从运行时链接面消失的恰好是这两个 .so。
   （注：ldd 取自本机 SKIP_GRPC、CUDA 12.1 构建产物。）
2. **libcuopt.so 真正的运行时链接面**（ldd 实证）：libcublas、libcublasLt、
   libcusparse、libnvJitLink、libamd+libsuitesparseconfig、libtbb、librmm、
   librapids_logger、libgomp（+libc/libstdc++/libm）。cudart 静态链接；
   curand/cusolver 在链接行声明但本构建被 as-needed 消除——应审计 cuDSS
   移除后是否已死，死则一并删除。
3. **CCCL 不是 RAPIDS，留在最终态。** 它是 NVIDIA/cccl 产品，目前经
   rapids-cmake 钉在 **3.4.0**；CTK 12.x 自带 CCCL 2.x、13.x 带 ~3.0/3.1，
   toolkit 自带版本不能直接替代——继续以平凡 FetchContent 拉 v3.4.0，并保留
   nvcc<12.4 的补丁（本机 CUDA 12.1 依赖它）。`cuda::std::span/mdspan`
   已验证存在于 CCCL 3.4.0，正好接替 `raft::device_span` 与少量 mdspan 视图。
4. **handle_t 实际只被抽取很小且固定的资源集**：`get_stream()`（~2,630 处）、
   `get_thrust_policy()`（474）、`sync_stream()`（249）、
   `get_cusparse_handle()`、`get_cublas_handle()`、批量 solve 中
   `resource::set_cuda_stream`（2 处）——**无 comms、无 raft stream pool、无
   device_properties 依赖**（comms 仅 1 个可删的测试行）。仓内已有原型：
   routing 的 `solution_handle_t`（cpp/src/routing/solution/solution_handle.cuh:26-78）。
5. **重型算法依赖为零**（已验证）：`raft::solver::LinearAssignmentProblem`、
   `raft::distance/matrix/cluster/neighbors/stats` 在 cuOpt 源码中零使用；
   cublas 实际只用 dot/nrm2/setpointermode 3 个函数；测试中无任何 raft matcher。
   cuOpt **已 vendoring 了逐位等价的 PCG 随机数 CPU 副本**
   （`cpp/src/utilities/pcgenerator.hpp`），47 处设备端 `PCGenerator` 只需加
   `__host__ __device__` 标注 + 改名。
6. **`thrust::device_vector` 不能替代 `device_uvector`**：构造/resize 会发
   value-init fill kernel、同步非池化 cudaMalloc、无 stream 参数、无
   `release()`。必须自写基于 per-device `cudaMemPool_t` + `cudaMallocAsync`
   的容器（~1.5–2k 行；RMM 为 Apache-2.0，可直接借鉴实现）。
7. **cuDSS 先例直接可复用**（本分支，40 文件 +2,847/−1,032）：单抽象点 →
   仓内替换 → 兼容性 no-op 参数 → 全量门禁（ctest 137/137、Python 108/108 +
   server 94、目标值等价 LP bitwise / QP ≤1e-9、不受影响路径对照组、性能门
   ≥50%，实测 0.77x–1.67x），记录于 `docs/dev/cudss_replacement_plan.md` 与
   `cudss_replacement_benchmark.md`。

## 4. 各依赖逐项分析

### 4.1 RMM（难度：最高的单项，但机械）

- **用了什么**：`device_uvector`（全部求解器状态的统一容器，2,207 处；~915 个
  stream-ordered `.resize()`、137 个 element 访问器、`release()` 在 Cython
  边界转移所有权，见 cpp/src/pdlp/solution_conversion.cu:36-114、
  cpp/src/routing/utilities/cython.cu:79-85）；`device_buffer`（96 处，Python
  边界类型）；`device_scalar`（204 处）；`cuda_stream_view`（310 处）；
  `exec_policy`（105 直接 + 474 个 `get_thrust_policy()` 间接，thrust 临时
  存储走 RMM 池）；owning `cuda_stream`/`stream_pool`（11 处）；
  `rmm::out_of_memory` 在 2 处**承担功能性回退**（feasibility_jump.cu:1014
  减半 climber 数重试；barrier.cu:4681 降级为 NUMERICAL_ISSUES）。
  库本身不设置 memory resource——池配置全在边缘，见 §5。
- **替换方案**：vendoring（RMM 为 Apache-2.0）API 兼容的
  `cuopt::device_uvector/device_buffer/device_scalar/stream` 类型（~1.5–2k
  LOC），底层换成 per-device `cudaMemPool_t` + `cudaMallocAsync/cudaFreeAsync`；
  自写 ~60–80 行 thrust 异步分配器包成 `cuopt::exec_policy`；OOM 转成独立的
  `cuopt::out_of_memory` 并清除 sticky error。
- **难度**：hard（体量）+ moderate（语义）。**风险**：stream-ordered 生命周期
  （批量 VRP 在 cython.cu:131-135 将 buffer 从已销毁的 pool stream 重新关联到
  调用方 stream——复刻错了就是 use-after-free）；thrust 临时分配的性能回退；
  OOM 语义静默漂移。
- ⚠️ **实施前需裁定**：`rmm::exec_policy` 的同步语义基座。源码中 `par_nosync`
  出现 0 次，RMM 维度调研告诫不要拿 `par_nosync` 做整体替换；实现 Phase 2 前
  必须对照 rmm 源码逐字确认后再定基座。

### 4.2 RAFT core（handle/span/copy/nvtx/宏/设备工具）

- **用了什么**：`handle_t`（资源集见 §3.4）；`device_span` 1,328 处；
  `raft::copy` 638 处；nvtx range 449 处；错误宏 ~770 处；
  `WarpSize/ceildiv/alignTo/swapVals/myAtomic*/laneId` 等 ~310 处；
  `warpReduce/blockReduce/blockRankedReduce` 39 处（树内已有注释记录
  raft::blockReduce 的 float-min bug，feasibility_jump_kernels.cu:766）。
- **替换方案**：`cuopt::handle_t`（~150–250 行：owning stream view + 惰性
  cublas/cusparse handle + `set_stream` 时重绑 `cublasSetStream/cusparseSetStream`，
  方法名保持一致使 ~3,500 调用点零改动）；`device_span` → `cuda::std::span`
  别名；`raft::copy` → 10 行 `cudaMemcpyAsync` 模板；nvtx → 直接 nvtx3
  （顺带删掉 get_raft.cmake 里的 nvtx 补丁）；宏 → `CUOPT_CUDA_TRY` 等
  （~100 行）；设备工具 → ~150–350 行 vendored header；
  **warp/block reduction 逐字移植 raft 实现**（保浮点归约顺序）。
- **难度**：moderate。**风险**：与 RMM 强耦合（`get_stream` 返回
  `rmm::cuda_stream_view`、`get_thrust_policy` 返回 `rmm::exec_policy`——
  不一起做就要把同一批文件改两遍，Phase 0 别名技巧一次性化解）；归约顺序影响
  PDLP 确定性模式与 MIP 启发式轨迹；cuopt_c.cpp 8 处和 barrier.cu 2 处的
  `raft::exception/cuda_error` catch 必须保持可区分。

### 4.3 RAFT primitives（计算原语）

- **用了什么**：cusparse 类型分发 shim 75 处（算法枚举均在调用点显式给出，
  wrapper 不含策略）；cublas shim 40 处（实际只有 dot/nrm2/setpointermode）；
  `raft::random::PCGenerator` 47 处（见 §3.5）；host 端 RNG（Philox/make_blobs）
  全部集中在合成数据生成器 generator.cu 一个文件；raft::linalg 逐元素操作
  ~40 处（直接映射 thrust::transform）。
- **替换方案**：~600–800 行 vendored header（镜像签名 + namespace sed），
  除数据生成器外**全部可做到 bit-exact**。
- **难度**：easy–moderate；**公共 API 零影响**（已验证 cpp/include 无任何
  raft::sparse/linalg/random/util 引用）。**风险**：生成器换 RNG 后固定 seed
  的合成数据集会变（仅测试/演示数据，需重生成 golden）；死 include 暴露的
  传递包含编译尾巴。

### 4.4 rapids-cmake + rapids-logger（构建基础设施）

- **用了什么**：rapids-cmake 在 configure 期从 GitHub 拉取（RAPIDS_BRANCH
  钉 release/26.06），提供 ~10 个功能（arch 初始化、版本头、`rapids_export`
  config 生成、CPM 版本钉、rapids-cython 等），从 3 个入口 include
  （cpp/CMakeLists.txt:14、python/cuopt/CMakeLists.txt:8、
  python/libcuopt/CMakeLists.txt:8）；**rapids-logger 是编译进 libcuopt.so 的
  运行时 .so**，但 cuOpt 侧 API 面极小——~677 个 `CUOPT_LOG_*` 调用点全走
  生成的宏，直接触库的只有 logger.{hpp,cpp} 等 4 个文件；
  `CUOPT_LOG_ACTIVE_LEVEL` 是 PUBLIC 编译定义（下游可见）。
- **替换方案**：删 cmake/RAPIDS.cmake、rapids_config.cmake、RAPIDS_BRANCH，
  换 ~150 行平凡 CMake + FetchContent（CCCL 3.4.0、GTest 1.16.0）；手写
  `cuopt-config.cmake`（rapids_export 的 find_dependency 链是主要正确性风险点，
  需消费方 `find_package(cuopt)` CI 测试）；logger 用 ~350–600 行 vendored
  实现（**level 枚举数值必须与 `RAPIDS_LOGGER_LOG_LEVEL_*` 完全一致**并随安装
  头发布，否则下游编译全断）；rapids-cython 直接 vendoring（自包含
  Apache-2.0，~200 行）；rapids-build-backend 换 scikit-build-core
  （**涉及 wheel 命名 -cu12/-cu13，是产品决策**）。
- **难度**：moderate。本维度成本（vendored logger 1.5–2 周、export config
  0.5–1 周、build-backend/wheel 命名 1–1.5 周、rapids-cython 0.5 周、deps
  清理 0.5 周）**无法被改名技巧压缩**，已计入 §7 总量。

### 4.5 Python 层（cudf / rmm / pylibraft / cupy / numba）

分两档：

- **LP/MILP：语义上今天就是 numpy 进 numpy 出**——cudf 只是内部胶水
  （solver_wrapper.pyx:61 的模块级 `import cudf` + 14 个
  `series_from_buf(...).to_numpy()` 转换），移除零 API 影响；C++ 已存在返回
  host `std::vector<double>` 的 `lp_cpu_solutions_t` 变体（solver.pxd:82-94）
  可直接复用。
- **Routing/distance_engine：cudf 就是公共 API**——DataModel setter 只接受
  cudf.Series/DataFrame（vehicle_routing_wrapper.pyx:62-78），Assignment 返回
  cudf 对象，~114 个 docstring 示例。这是**对所有 routing 用户的破坏性变更**，
  需要产品决策（选项见 §8）。
- server 的 cudf 仅是 JSON→device ETL；rmm 在 server 仅做启动时建池（见 §5）；
  pylibraft 仅为 cimport handle 类型，从未暴露给用户；numba/cupy 大半是死
  import，可低成本清理；cuopt_sh_client 零 RAPIDS 依赖。
- **强顺序依赖：必须等 C++ 后继类型落地。**

## 5. 内存资源层专项：`cuda::mr::*` / `rmm::mr::*` 可以完全消失

一个容易高估的点：`cuda::mr::*`（libcu++ 的 memory-resource 概念层，
`async_resource_ref`、`any_resource`、property 机制等）看起来「重度使用」，
但**重度使用发生在 RMM 自身的实现里**（rmm 自迁移到 cuda::mr 概念后，其
resource_ref 机制建立在 libcu++ 之上）。cuOpt 自身的真实暴露面经穷举确认
只有以下几处：

| 位置 | 内容 |
|---|---|
| cpp/tests/utilities/base_fixture.hpp:64 | **全仓唯一一处直接 `cuda::mr::` 使用**：`cuda::mr::any_resource<cuda::mr::device_accessible>` 作 gtest fixture 的类型擦除返回值；fixture 支持 `--rmm_mode` 选 binning/cuda/pool/managed（默认 pool） |
| cpp/cuopt_cli.cpp:421-431 | 每 GPU 一个 `rmm::mr::cuda_async_memory_resource` + `set_per_device_resource` |
| cpp/tests/mip/{multi_probe,load_balancing}_test.cu | 各 1 个 async resource + `set_current_device_resource` |
| python/cuopt_server/.../solver.py:373-377 | worker 启动时 `rmm.mr.PoolMemoryResource` + `set_current_device_resource` |
| python/cuopt_server/.../routing/solver.py:123-125 | `isinstance` 内省 resource 类型（StatisticsResourceAdaptor / PoolMemoryResource，用于内存记账） |

**库代码（cpp/src、cpp/include）零显式 memory-resource 使用**——所有
`device_uvector` 分配都走隐式的「当前设备资源」。也就是说 cuda::mr 提供的
可组合、可插拔抽象在 cuOpt 里实际只用到一种形态：**进程启动时选一次分配器**。

**结论：rmm 替换完成后，整个 cuda::mr 概念层自动消失，无需任何替代品。**
（即使保留也不构成 RAPIDS 依赖——cuda::mr 属于 CCCL——但 shim 方案下它
直接变成无关项。）替代映射：

- 库内分配 → `cuopt::device_uvector` 直接走 per-device `cudaMemPool_t`
  （CUDA runtime 原生 API），不需要 resource_ref 间接层；
- CLI 的 per-GPU async resource → 每 GPU `cudaDeviceGetDefaultMemPool` /
  `cudaMemPoolCreate` + `cudaDeviceSetMemPool`；
- pool 语义（server/测试默认）→ `cudaMemPoolSetAttribute(ReleaseThreshold=UINT64_MAX)`
  + 可选预热分配模拟 initial_pool_size；
- 测试 fixture 的 4 种模式 → pool/cuda 模式映射 release-threshold 配置，
  managed 模式走 `cudaMallocManaged` 小分支；binning 模式建议直接删除
  （建立在 async pool 之上收益存疑，且非默认模式）；`any_resource` 类型擦除
  → fixture 内部一个小 variant 即可；
- server 的内存记账内省 → `cudaMemPoolGetAttribute`
  （UsedMemCurrent/ReservedMemHigh 等）经小型 C API 暴露给 Python；
- 若未来确需可插拔分配器，~100–200 行的自有虚接口（3 个实现：async-pool /
  managed / 同步 cudaMalloc 回退）即可，**不需要复刻 cuda::mr 的编译期
  property/概念机制**——那套机制服务于 cuOpt 从未使用的静态组合场景。

**注意事项**：`cudaMallocAsync`/`cudaMemPool_t` 在部分 legacy/vGPU 配置不可用
（`cudaDevAttrMemoryPoolsSupported` 探测），需保留同步 cudaMalloc 回退路径
（已列入 §10 风险 6）。

## 6. 推荐路线（混合策略 + Phase-0 别名技巧）

**策略**：存储/handle/span/copy/nvtx/error 层做 API 镜像 shim，计算原语做
惯用法替换；明确不推荐全惯用法重写（估 24–48 人周、高回归风险、终态无额外
收益）。核心是 **Phase 0 先落「纯别名头」**——`cuopt/cuda/*.hpp` 中
`using device_uvector = rmm::device_uvector<T>;` 等，脚本化 sed 把 ~6,650 个
调用点改成 `cuopt::` 拼写。**构建全程保持绿色**（rmm/raft 仍在链接），巨型
diff 语义惰性、可按生成物审查；之后逐层换实现，最后一步才真正切断依赖。

| 阶段 | 内容 | 工期 |
|---|---|---|
| **Phase 0** | 别名头 + 脚本化全量改名（零行为风险，可作 go/no-go 探针；尽早落 upstream 减少合并冲突） | 1 周 |
| **Phase 1** | 叶子 shim 换实现：span→`cuda::std::span`、copy、nvtx3、错误宏、device utils、bit-exact PCG 移植、linalg→thrust、cublas/cusparse 薄 wrapper | ~2 周 |
| **Phase 2** | **内存层（最高风险）**：device_buffer/uvector/scalar 落到 per-device `cudaMemPool_t`；`cuopt::exec_policy` 自定义异步分配器**与 policy 切换同一 PR 落地**；测试 fixture、cuopt_cli 多 GPU 池（§5 映射） | 2–3 周 |
| **Phase 3** | `cuopt::handle_t` + 22 个公共头迁移 + Cython handle/buffer 重做 | 2 周 |
| **Phase 4** | 构建系统：删 rapids-cmake/rmm/raft/rapids-logger 拉取，平凡 CMake + 手写 cuopt-config，vendored logger，rapids-cython vendoring，build-backend 切换，dependencies.yaml/conda/wheel 清理 | 3–5 周 |
| **Phase 5** | 验证战役（见 §9） | 1–2 周 |

期间用 `CUOPT_USE_RAPIDS_SHIMS=ON/OFF` CMake 开关让两套后端在 Phase 1–3 都可
构建，便于二分定位。Python/Cython 严格排在 C++ 后继类型之后；cudf routing
迁移和 CI 脱钩作为独立后续工作流。

## 7. 工作量汇总

| 维度 | 单独估计 | 说明 |
|---|---|---|
| RMM | 5–7 人周 | 含 vendored 容器库、~3,000 点迁移、Cython 边界、性能验证 |
| RAFT core | 5–7 人周 | 含 handle、770 宏点、reduction 移植 + 解质量回归 |
| RAFT primitives | 2.5–4 人周 | 主要成本是验证而非编码 |
| 构建基础设施 | 6–9 人周 | 不含 CI 脱钩（+2–3，可推迟） |
| Python | 7–10 人周 | 其中 routing cudf 迁移占 4–5 |
| 逐项相加 | 26–37 人周 | 各维度独立含验证/排序开销，存在大量重复计入 |
| **集成估计（推荐口径）** | **16–24 人周（全量，含 cudf）** | Phase-0 别名技巧把机械迁移做成一次脚本动作、验证战役跨维度共享；但构建/打包维度的 logger/export/wheel 项不可压缩，已计入 Phase 4 的 3–5 周 |

锚点：cuDSS 移除（需发明新求解器）40 文件、~2–4 周提交跨度；本工作文件量
~14 倍但零新算法、>75% 调用点集中于 5 个符号。

## 8. 公共 API 影响

| 表面 | 影响 | 选项 |
|---|---|---|
| **C++ SDK** | **破坏性**：22 个安装头暴露 raft/rmm 类型——`raft::handle_t*` 出现在 solve.hpp:73,109,142,148、optimization_problem.hpp:118 等签名中；`rmm::device_uvector&` 出现在 optimization_problem_interface.hpp:273-276 的**纯虚 getter**中；solver_settings 公共头以 `rmm::cuda_stream_default` 作默认实参（solver_settings.hpp:53,56,83 等），pdlp/solver_settings.hpp:190-192 甚至在默认实参里构造 `rmm::device_uvector` 临时对象 | 重命名为 `cuopt::` 类型，签名形状不变，下游迁移≈改名；提供 `using` 过渡别名 + cudaStream_t 转换构造；按 major 级别发布说明 |
| **C API** | **零影响**（cuopt_c.h 无任何 raft/rmm） | 无需动作 |
| **Cython 内部** | 8 文件 pylibraft cimport、41 个 DeviceBuffer 转移点，用户不可见 | 优先：C++ 改返回 host 向量（LP 侧本就立即转 host）；若需保留 device 返回：~100–150 行自有 buffer 类实现 `__cuda_array_interface__`（保 cupy/numba/torch 零拷贝互操作） |
| **Python LP/MILP** | **零语义影响**（已是 numpy 进出） | 删内部 cudf 胶水即可 |
| **Python routing/distance_engine** | **重大破坏**：cudf 是文档化的输入/输出类型 | (a) 硬切 numpy/pandas + major 版本；(b) 鸭子类型经 `__cuda_array_interface__` 继续接受 cudf 而不 import 它；(c) cudf 降级为可选软依赖。需产品决策 |
| **打包** | wheel 依赖集缩水（cudf 是最重的传递依赖，拉 arrow 等，移除是最大安装体积收益）；弃 rapids-build-backend 改变 -cu12/-cu13 命名机制 | 用户安装时可见 |
| **Server REST/gRPC** | 协议零影响 | 仅内部 ETL 改写 + 池初始化/内存记账换新 API（§5） |

## 9. 验证方法（镜像 cuDSS 移除门禁）

- 每阶段全量 ctest（137 项）+ Python 套件（108 项 + server 94）；
- 目标值等价 vs 移除前 RAPIDS 基线 wheel（LP bitwise、QP ≤1e-9 风格）；
- 不受影响路径对照组（如 dual_simplex）；
- **显式分配器性能门**——按 `docs/dev/cudss_replacement_benchmark.md` 的 MPS
  实例基准对比池化基线，重点压 routing 批量 solve、feasibility_jump（每迭代
  thrust 临时分配 23 处）、PDLP；
- PCG **序列等价单元测试**在切换前先落地；
- compute-sanitizer 跑 routing+pdlp 冒烟集（确定性目标值门禁抓不到竞态）；
- 多 GPU 路径（cuopt_cli per-device 池、PDLP 并发 climber）专项验证；
- 必要时 nsys 驱动优化（cuDSS 先例做过 3 轮）。

## 10. 风险清单（Top 10）

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **Stream-ordered 生命周期 bug**：批量 VRP 把 buffer 从已销毁 pool stream 重关联到调用方 stream（cython.cu:131-135）；shim 用错 stream 释放即 use-after-free | 逐行对照 rmm 源码评审 shim；compute-sanitizer 纳入门禁；批量 solve 专项测试 |
| 2 | **thrust 临时存储性能回退**：105+474 个 policy 点现走池；裸 `par.on(stream)` 在热循环引入同步 cudaMalloc | 自定义 cudaMallocAsync 分配器与 policy 切换同 PR 原子落地；基准对比；先裁定 sync/nosync 基座（§4.1） |
| 3 | **RNG 轨迹改变**：47 个 PCGenerator 点驱动启发式搜索 | 利用已 vendored 的 pcgenerator.hpp 做 bit-exact 替换；序列等价单测先行；严禁「顺手改进」 |
| 4 | **浮点归约顺序**：blockReduce/blockRankedReduce 39 点影响 PDLP 确定性模式与 MIP 轨迹 | 逐字移植 raft 实现而非换 cub；解质量回归跑，不止单测 |
| 5 | **OOM 回退语义**：2 处 catch `rmm::out_of_memory` 承担功能（减半 climber / 降级状态） | 专用 `cuopt::out_of_memory` 类型 + 清 sticky error；内存受限 GPU 专项测试 |
| 6 | **池行为与内存记账差异**：server 显式建池并内省 resource 类型；cudaMallocAsync 在部分 legacy/vGPU 不可用 | release threshold=UINT64_MAX + 预热；`cudaMemPoolGetAttribute` 替代记账；保留同步 cudaMalloc 回退路径 |
| 7 | **公共 C++ API 破坏**（22 头文件） | 过渡别名 + 转换构造；major 级发布说明；C API/server 用户天然免疫 |
| 8 | **routing Python cudf 破坏** | 提前产品决策；鸭子类型 shim 沿用 cuDSS 的兼容姿态；9 个测试文件与 API 同一变更内迁移 |
| 9 | **规模/合并冲突**：370–550 文件，上游活跃（QCQP/SOCP 刚落地） | Phase-0 别名 diff 尽早落地；脚本化、可重放的 sed 提交；长寿分支最小化 |
| 10 | **构建/打包长尾**：rapids_export 的 find_dependency 链、logger level 宏数值、CUDA arch 列表（'RAPIDS' 魔法值）、wheel 命名 | 消费方 find_package(cuopt) CI 测试；cuobjdump 对比 arch 覆盖；logger 数值钉死在安装头；CI 脱钩明确推迟 |

## 11. 长尾遗漏面（交叉审查补充，均可推迟但须列入收尾清单）

- `regression/`：夜间回归基建依赖 RAPIDS 内部工具 rapids-mg-tools
  （routing/lp/mip_regression_test.sh、cronjob.sh 均 source
  `$RAPIDS_MG_TOOLS_DIR/script-env.sh`；config.sh 有 `WORKER_RMM_POOL_SIZE=24G`）；
- `docs/` Sphinx 树：routing 示例直接 `import cudf`
  （tsp_batch_example.py、intra_factory_example.py）；faq.rst:125 指引用户排查
  rmm 错误；`_static/install-selector.js:78-90` 硬编码 rapidsai-wheels-nightly
  pip index 与 `-c rapidsai` conda channel——**移除后若不改，安装文档会指向
  失效命令**；
- `.pre-commit-config.yaml`：rapidsai/dependency-file-generator hook 与
  rapidsai/pre-commit-hooks；
- `conda/environments/all_cuda-*.yaml`（4 个生成文件）与
  conda/recipes（cuopt/recipe.yaml:77-102 pin pylibraft/rmm/cudf/
  rapids-build-backend）。

## 12. 移除后的最终依赖清单

- **CUDA Toolkit**：cublas/cublasLt、cusparse、nvJitLink、nvtx3（header）、
  静态 cudart；curand/cusolver 待审计（本构建 ldd 缺席，可能随 cuDSS 移除已死）
- **CCCL 3.4.0**（thrust/cub/libcu++，平凡 FetchContent，带 nvcc<12.4 补丁）——
  设计保留；其中 `cuda::mr` 概念层不再被引用（§5）
- **系统库**：OpenMP、TBB（papilo 需要）、SuiteSparse AMD（cuDSS 移除引入）、
  bzip2/zlib（可选，MPS 压缩）
- **FetchContent 求解器依赖**：papilo（钉 SHA，需 Boost+quadmath）、
  PSLP v0.0.8、dejavu v2.1、argparse v3.2；测试 gtest 1.16.0
- **可选 server 栈**：gRPC、protobuf、OpenSSL、libuuid
- **新增自有代码**：~3,500 行 shim（容器/handle/span/wrapper/RNG/logger）+
  ~1,500 行 shim 单测，全部 Apache-2.0 vendoring 许可干净；
  thirdparty/THIRD_PARTY_LICENSES 相应更新
- **消失项**：librmm.so、librapids_logger.so、libraft-headers、pylibraft、
  rmm(py)、cudf、rapids-cmake、rapids-build-backend、rapidsai conda channel 与
  rapids 轮子 nightly index、RAPIDS_BRANCH——版本号与 RAPIDS 双月发布列车脱钩
  （nightly 同列车依赖破裂这一常发故障源随之消除）
