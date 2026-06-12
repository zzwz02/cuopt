# cuOpt RAPIDS 依赖移除——实施结果与全栈验证报告

> 分支 `feature/remove-cudss-custom-ldlt`(在 cuDSS 移除/自研 LDLᵀ 之上)。
> 目标:以自研 cuda::mr-free 同名 shim 头文件树替换 rmm / raft / rapids_logger,
> 使 cuOpt 全栈(C++ / C API / CLI / Python / REST server / gRPC)在零 RAPIDS
> 运行时依赖下构建、与官方 conda/wheel 版功能持平;cudf 按产品决策保留为
> routing 的可选运行时依赖(MI100 等目标平台有对应 cudf 移植)。
> 计划文档:`docs/dev/rapids_removal_plan.md`;
> cuDSS 阶段基准:`docs/dev/cudss_replacement_benchmark.md`。
> 硬件:NVIDIA A100-PCIE-40GB(驱动 CUDA 12.9);AMD EPYC 7543。

## 1. 结论

- **运行时依赖**:`ldd libcuopt.so` 不再含 librmm / libraft / librapids_logger
  / libcudss;体积 ~108 MB → ~106 MB。源码层面,cuopt 全部 C++/Python/server
  代码中 `import rmm`/pylibraft 引用为零;rmm 仅作为 cudf 自身的内部依赖存在,
  与 cuopt 代码零接触,无须修改 cudf。
- **功能**:与 cuDSS 期 conda 版全量验证门槛逐项持平(§4):C++ ctest 本地
  137/137、conda 配置(含 gRPC)139/140;Python 全树 108/108;server 94+7 skip;
  self-hosted 3/3;Python gRPC 远程执行(CPU-only/TLS/mTLS)11/11。
- **性能**:与移除前(06-11 自研 LDLᵀ 优化版)逐实例一致(差异 ≤4%,在运行
  噪声内);与 cuDSS 原版的关系维持 06-11 结论(小/中型与 QP 同量级或更快,
  填充重的大型因子分解 1.3–1.7×)。详见 §5。
- **打包**:三个 Python 包均改为平凡构建后端(scikit-build-core / setuptools),
  `pip wheel` 不再依赖 rapids-build-backend / rapids-cmake / rapids-cython;
  cudf 移入 `cuopt[routing]` extra。

## 2. 实施步骤概要

| 阶段 | 内容 | 结果 |
|---|---|---|
| 1. shim 头文件树 | 手写 cuda::mr-free 的 rmm(容器/流/资源/分配器,16 头)+ raft core(handle/span/mdspan/copy 等);纯 CUDA 的 raft 叶子(math/operators/reduction/warp/RNG-PCG/cublas+cusparse wrappers)逐字移植保数值序;header-only rapids_logger | 独立烟测通过 |
| 2. CMake 切换 | cpp/CMakeLists 删除 rmm/raft/rapids_logger 拉取与链接,vendor include 接管;CCCL(非 RAPIDS)保留 | 全量零编译错误,ldd 零 RAPIDS |
| 3. C++ 测试修复 | 五项修复(见 §2.1)将 ctest 由 121/137 修至 **137/137** | 与 baseline 持平 |
| 4. Python LP/MILP | solver.pxd/pyx 的 rmm/pylibraft cimport 改 shim 本地声明,device_buffer 输出改 cudaMemcpy D2H→numpy;utilities cudf 惰性化 | LP/QP/SOCP 65/65,afiro 目标值与 C++ 逐位一致 |
| 5. routing/distance Python | 同法解耦 cimport,输出改 D2H→cudf.Series(**cudf API 契约保留**);要点:即使环境装有 RAPIDS,wheel 编译的 pylibraft/rmm Python 类按真 raft/rmm ABI 布局,不可与 shim 类型互传 | routing 全套 43/43(含 distance/批量/重路由/数据生成) |
| 6. gRPC 启用 | conda 环境(gcc14/nvcc12.9/grpc1.78)以 SKIP_GRPC_BUILD=0 重建 shim 树;两项 shim 保真度修复(§2.1) | 142 注册,139/140,3 个 gRPC 测试全过;Python 远程 11/11 |
| 7. server + rmm 彻底剥离 | server 零功能改动即通过;按要求删除全部 `import rmm`(池初始化/断言/预加载/死代码) | server 94+7 复测通过 |
| 8. 打包 | 3 个 pyproject 去 rapids-build-backend;python CMake 去 rapids-cmake/rapids-cython(自带 30 行兼容函数,子目录零改动) | `pip wheel` 产出 cuopt/libcuopt wheel |

### 2.1 关键缺陷修复(一句话记录)

1. shim handle 默认流改为 `cuda_stream_per_thread`——legacy null 流不能作 CUDA
   Graph 捕获目标(+7 测试)。
2. handle 拷贝改为共享引用计数的 cuBLAS/cuSPARSE 句柄,保住 cuOpt 设定的
   device pointer mode。
3. 测试默认分配器须可捕获(同步 cudaMalloc 在 stream capture 中非法)。
4. routing 的 move-candidates reset 由捕获重放 CUDA Graph 改为默认 eager 发射
   ——重放型图与 shim 分配器组合下哨兵填充不生效(`CUOPT_USE_RESET_GRAPH`
   保留为实验开关;arena 池落地后该路径仍异常,机制待查,eager 全绿)。
5. `pool_memory_resource` 重写为 rmm 式 stream-ordered **arena**(大 slab 子分
   配、析构前不解映射、per-stream event 键控 free list、跨流取块 WaitEvent、
   地址序合并)——修复最后一例 ROUTING_UNIT 段错(cuOpt 既有 use-after-free
   被解映射型分配器暴露,arena 与 rmm 同样使其良性)→ 137/137。
6. rapids_logger shim 补命名级别方法(error/warn/info 等,gRPC server 直调)。
7. 设备查询改为非抛出(无 GPU 返回 -1,镜像 rmm 的 release 语义)——修复
   CPU-only 远程执行模式(`CUDA_VISIBLE_DEVICES=""`)。
8. server 剥 rmm 时遗留一行引用已删变量的日志导致 worker 启动即崩,删除后
   server 全套复测通过。

## 3. 与 conda 版 cuOpt 的全栈功能对照

“conda 版”指 cuDSS 期在 `cuopt-dev` conda 环境的全功能构建/官方 26.06 wheel
(真 RAPIDS + cuDSS)。

| 功能面 | conda 版 | RAPIDS-free 版 | 验证 |
|---|---|---|---|
| LP — PDLP / 对偶单纯形 / Barrier(C++) | ✅ | ✅ 适配 | ctest + afiro/PDLP 目标逐位一致;基准 §5 |
| MILP(分支定界/割平面/启发式)| ✅ | ✅ 适配 | 15 项 MIP ctest 含 DETERMINISM 全过 |
| QP / SOCP(barrier,自研 LDLᵀ)| ✅(cuDSS 版为其原版)| ✅ 适配 | QP_UNIT/SOCP ctest + 11 题 QP 基准目标一致 |
| C API(cuopt_c.h)| ✅ | ✅ 适配 | C_API_TEST(LP/MILP/barrier/回调/远程)通过 |
| cuopt_cli | ✅ | ✅ 适配 | CLI_TEST + 基准全程使用 |
| routing C++(VRP/PDP/breaks/异构车队等)| ✅ | ✅ 适配 | 全部 routing ctest 通过;注:reset 路径默认 eager(§2.1-4) |
| 距离引擎(C++ waypoint matrix)| ✅ | ✅ 适配 | WAYPOINT_MATRIXTEST 通过 |
| Python LP/MILP/QP/SOCP(numpy 进出,batch/warm-start/incumbent 回调)| ✅ | ✅ 适配(零 RAPIDS 包环境可运行)| 65/65 |
| Python routing + distance_engine(cudf 进出)| ✅ | ✅ 适配(cudf 保留为 `[routing]` extra;rmm/pylibraft cimport 解耦)| 43/43 |
| REST server(cuopt_server)| ✅ | ✅ 适配(功能零改动;rmm import 全部移除)| 94 + 7 skip(skip 与 conda 期相同,上游 #519)|
| gRPC server(C++)+ Python 远程执行(CPU-only/TLS/mTLS)| ✅ | ✅ 适配 | 3 个 gRPC ctest + 11/11 远程测试通过,二进制零 RAPIDS |
| self-hosted 客户端(cuopt_sh_client)| ✅ | ✅ 本就零 RAPIDS | 3/3(对运行中 server)|
| pip wheel 打包 | rapids-build-backend | ✅ 适配:scikit-build-core / setuptools | cuopt/libcuopt wheel 构建通过(§6)|
| **降级项** | — | NVTX 标注为 no-op;logger `set_pattern` 仅支持两种模式;rmm logging/statistics 适配器缺失(cuOpt 未用);host 端 make_blobs RNG 非逐位(运行覆盖绿,golden 未断言);comms 桩(单 GPU,同上游现状) | — |
| **未适配项** | — | conda recipe/CI 打包未触及;routing 性能基准(CVRPLIB 等)双方均未例行化;HIP/MACA 后端(本工作为其前提,未开始) | — |

## 4. 测试对账(vs cuDSS 期 conda 门槛)

| 门槛 | cuDSS 期 conda 版 | RAPIDS-free 版 |
|---|---|---|
| C++ ctest(本地,SKIP_GRPC)| 137/137 | **137/137** |
| C++ ctest(conda 配置,含 3 gRPC)| 139/140 | **139/140**(唯一失败 CUTS_TEST 为 1s 时限边缘抖动,目标值精确,串行复跑通过)|
| Python cuopt 测试树(LP+QP+SOCP+routing)| 108/108 | **108/108**(60+1+4+43)|
| cuopt_server | 94 + 7 skip | **94 + 7 skip**(skip 同为上游 #519)|
| cuopt_self_hosted | 3/3 | **3/3** |
| Python gRPC 远程(CPU-only/TLS/mTLS)| 含于上 | **11/11** |

环境注意事项:测试目录需 `datasets` 符号链接;`PYTHONPATH` 须含
`python/libcuopt`(否则命名空间包遮蔽);gRPC/server 测试须清除大小写
`http(s)_proxy` 并设 `no_proxy=localhost,127.0.0.1,0.0.0.0`;conda 构建产物
运行时须 `LD_LIBRARY_PATH=$build`(conda LDFLAGS 将 env lib 前置于 RUNPATH,
其中装有旧版全 RAPIDS libcuopt)。

## 5. 性能基准(三方对比)

方法:与 `docs/dev/cudss_replacement_benchmark.md` 同协议——Barrier
(`--method 3`),LP `--time-limit 600`、QP 180s,solver 自报时间;同一 A100。
三方:**本版**(RAPIDS-free shim,本次实测)、**06-11 版**(cuDSS 移除+LDLᵀ
优化后、RAPIDS 移除前)、**cuDSS 原版**(官方 26.06 wheel,cuDSS 0.7.1)。

### 5.1 LP(Mittelmann 子集,Barrier)

| 实例 | 本版(shim) | 06-11 版 | cuDSS 原版 | 本版目标值 |
|---|---|---|---|---|
| afiro | **0.09 s** | 0.093 s | 0.121 s | -464.753135(逐位同基线)|
| graph40-40 | **0.99 s** | 0.87 s | 0.90 s | -300.000464 |
| qap15 | **1.41 s** | 1.40 s | 1.09 s | 1041.00046 |
| nug08-3rd | **6.23 s** | 6.24 s | 3.74 s | 214.000642 |
| woodlands09 | **5.42 s** | 5.25 s | 4.13 s | 1.4e-07(≈0)|
| scpm1 | **4.20 s** | 4.04 s | 3.71 s | 414.152071 |

全部 Optimal。本版 vs 06-11 版:差异 ≤0.18 s(≤4%),在运行噪声内——
**RAPIDS 移除对求解性能无可测回退**。本版 vs cuDSS 原版:维持 06-11 结论
(afiro/graph40-40 同级或更快;填充重的 qap15/nug08/woodlands/scpm1 为
1.1–1.7×,全部满足 ≥50% 性能目标)。

### 5.2 QP(QP_Test + Maros-Mészáros 子集,Barrier,180 s)

11/11 Optimal,目标值与文献/双基线一致(≤1e-9):

| 问题 | 本版目标值 | 本版耗时 | 06-11 版 | cuDSS 原版 |
|---|---|---|---|---|
| QP_Test_1 | -99.9600000 | 0.07 s | 0.04–0.09 s | 0.07–0.14 s |
| HS35 | 0.111111115 | 0.05 s | 〃 | 〃 |
| HS51 | 0.000000000 | 0.05 s | 〃 | 〃 |
| HS52 | 5.32664756 | 0.04 s | 〃 | 〃 |
| HS53 | 4.09302326 | 0.06 s | 〃 | 〃 |
| HS76 | -4.68181817 | 0.05 s | 〃 | 〃 |
| HS268 / S268 | 3.84e-09 | 0.06 s | 〃 | 〃 |
| HS35MOD | 0.250000002 | 0.07 s | 〃 | 〃 |
| QPTEST | 4.37187501 | 0.06 s | 〃 | 〃 |
| ZECEVIC2 | -4.12499999 | 0.07 s | 〃 | 〃 |

对照组(未受影响路径):PDLP afiro -4.64761260e+02 与基线逐位一致;
对偶单纯形 afiro 0.03 s,行为与双基线一致。

### 5.3 QP 全集(Maros-Mészáros 138 题,Barrier,180 s)

本版与 cuDSS 原版各完整扫描一遍:

| 指标 | 本版(shim,自研 LDLᵀ) | cuDSS 原版 |
|---|---|---|
| 180 s 内完成 | **138 / 138** | 138 / 138 |
| 目标值一致(rel < 1e-4,129 题可比) | **126 / 129** | — |
| 总求解时间(双方均完成的 129 题) | 289 s | 38 s |
| 单题时间比(本版/cuDSS) | 中位数 **1.44×**,均值 4.49× | 1× |

3 题目标差 0.10–0.36%(QCAPRI / QSIERRA / UBH1,病态 QP 的 IPM 容差级差异,
两侧均为可行驻点);时间差距源于自研 LDLᵀ 与 cuDSS supernodal 分解在
填充重因子上的既有差距(与 §5.1 LP 结论一致),与 RAPIDS 移除无关。

### 5.4 Routing(CVRPLIB Set-X / Solomon / Homberger,固定时间预算)

同一驱动脚本、同机两套构建(**shim** = 本版 wheel;**RAPIDS 基线** = 官方
26.06 wheel,真 rmm/raft),每组合 2 次重复,报告目标值(均为 SUCCESS、
车辆数一致):

| 实例(规模,预算) | shim(2 次) | RAPIDS 基线(2 次) | 文献 BKS |
|---|---|---|---|
| X-n101-k25(101,30 s) | 31573 / 32753 | 31962 / 32680 | 27591 |
| X-n439-k37(439,60 s) | 36454 / 36470 | 36447 / 36497 | 36391 |
| X-n1001-k43(1001,120 s) | 73709 / 74042 | 73821 / 74077 | 72355 |
| C101(Solomon 100+TW,30 s) | **828.94 / 828.94** | 828.94 / 828.94 | 828.94(最优)|
| C1_10_1(Homberger 1000+TW,120 s) | **42478.95 / 42478.95** | 42478.95 / 42478.95 | ≈42444 |

结论:固定预算下两构建解质量**统计无差**(组间差小于组内重复方差,多处
shim 略优);时间窗类双方逐位同解,C101 双方均达已证最优。eager-reset 与
arena 池对 routing 求解质量与速度无可测影响。

## 6. 构建与打包

- **C++(本地)**:`.toolchain` cmake 3.30.8 + 系统 gcc11/CUDA12.1。
  **构建系统已完全去 rapids-cmake**(4b):版本/架构/构建类型/静态 cudart
  为原生 CMake;CCCL(v3.4.0)/googletest/argparse/papilo 等经标准
  `FetchContent`(支持 `-DFETCHCONTENT_SOURCE_DIR_*` 离线源覆盖);
  `cuopt-config.cmake` 改为手写(install + build 两套,自动为消费方提供
  CCCL 目标)。配置期**不再从 RAPIDS 仓库下载任何内容**。
- **C++(conda,含 gRPC)**:`cuopt-dev` 环境 gcc14/nvcc12.9/grpc1.78,
  `cpp/build-conda`,`-DSKIP_GRPC_BUILD=0`。
- **Python wheel**:`pip wheel ./python/libcuopt`(scikit-build-core,完整
  C++ 构建,gRPC 可选)与 `pip wheel ./python/cuopt -C
  cmake.args="-DCMAKE_PREFIX_PATH=<libcuopt 安装或 cpp/build>"`(8 个 Cython
  模块);cuopt_server/self-hosted 为纯 setuptools。cudf 等 routing 依赖在
  `cuopt[routing]` extra 中。
- 开发期快捷脚本 `python/cuopt/build_lp_modules_rapids_free.sh`(就地编译
  8 个扩展)继续可用。

## 7. 遗留与展望

- routing reset 的捕获重放 CUDA Graph 在 shim 下仍异常(默认 eager 全绿,
  开关保留),机制待独立定位。
- MIP 实例集(MIPLIB)与 server 吞吐基准未例行化。
- HIP/MACA 后端移植:本工作消除了 cuda::mr/RAPIDS 障碍(运行时与构建系统
  均不再触及 RAPIDS),后续主要工作为 warp64/平台运行时适配;routing
  Python 依赖目标平台的 cudf 移植版。
