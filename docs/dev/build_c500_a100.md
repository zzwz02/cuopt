# cuOpt 源码构建与测试指南（MetaX C500 / NVIDIA A100）

> 本文说明在**同一台机器**上从源码构建 cuOpt,并运行 `docs/dev/maca_c500_results.md` 第 1 节功能矩阵的全部测试。
> - **C500**:经 MACA / cu-bridge(warp64),构建目录 `cpp/build_maca`,开关 `-DCUOPT_MACA=ON`。
> - **A100**:标准 NVIDIA CUDA 12.9 路径,构建目录 `cpp/build_cuda`,`CUOPT_MACA` 默认 `OFF`。
>
> 两个构建目录互不影响,可并行构建(本机 128 核 / 503 GB,同时编译无压力)。所有命令均经实测,默认从**仓库根目录**执行。

---

## 0. 环境与依赖安装

cuOpt 不依赖系统 RAPIDS;C++ 核心 / C API / CLI 经 vendored rmm/raft/rapids_logger shim 构建。下列组件本机已就绪;从零搭建时按各小节安装。

### 0.1 硬件与驱动

| 平台 | 驱动 / 运行时 | 校验 |
|---|---|---|
| MetaX C500 | MACA 3.0.x + MetaX KMD(`/opt/mxdriver`) | `mx-smi` 显示 1 块 C500 |
| NVIDIA A100 | NVIDIA 驱动 + CUDA 12.9(`/usr/local/cuda`) | `nvidia-smi` 显示 1 块 A100 |

### 0.2 MACA / cu-bridge(C500 必需)

MetaX 提供,安装于 `/opt/maca`(含 `tools/cu-bridge` 的 `cmake_maca`/`ninja_maca`/`pre_make`、`mxgpu_llvm/bin/mxcc`、`lib/libmc*.so`)。`source maca_env.sh` 导出 `MACA_PATH`、cu-bridge `PATH`、`MACA_DIRECT_DISPATCH=1` 等。校验:`cmake_maca --version`、`command -v mxcc`。

### 0.3 NVIDIA CUDA Toolkit(A100 必需)

`/usr/local/cuda`(本机 12.9)。校验:`nvcc --version`。

### 0.4 通用构建工具链(`.toolchain/`,两平台共用)

| 组件 | 路径 | 获取方式 |
|---|---|---|
| CMake 3.30.8 | `.toolchain/cmake-3.30.8-linux-x86_64` | 官方二进制包(cuOpt 需 ≥ 3.30,系统 3.22 不够):`curl -LO https://github.com/Kitware/CMake/releases/download/v3.30.8/cmake-3.30.8-linux-x86_64.tar.gz && tar xf` |
| Boost 1.84 | `.toolchain/boost-install` | papilo 依赖;`b2 install --prefix=.toolchain/boost-install`(已编译) |
| oneAPI TBB 2021.13 | `.toolchain/oneapi-tbb-2021.13.0` | papilo 依赖;解压官方 oneTBB 包 |
| CCCL 3.4.0 源 | `.toolchain/cccl-3.4.0-src` | **A100 必需**:`git clone https://github.com/NVIDIA/cccl && git checkout <3.4.0 commit>`(公网无 `v3.4.0` 正式标签,见 §2 注 2) |
| BZip2 | `/opt/conda`(`BZIP2_ROOT`) | papilo 依赖;系统无 libbz2-dev,用 conda 内 libbz2 |

其余 C++ 依赖(papilo / pslp / dejavu / argparse / googletest)在首次 configure 时由 CMake `FetchContent` 从公网 GitHub 拉取,**需联网**。

### 0.5 Python 环境与依赖

基础:conda(`/opt/conda`,python 3.10)。构建 RAPIDS-free 扩展仅需:

```bash
/opt/conda/bin/python -m pip install cython numpy          # 本机已装(cython 3.2.5 / numpy 1.26.4)
```

**routing/distance 的 Python 路径(C500)** 另需 MetaX mcdf 栈(cudf/cupy/numba 的 MACA 实现)。一键安装(详见 §5):

```bash
source maca_env.sh && export PATH=/opt/conda/bin:$PATH
bash python/cuopt/setup_maca_rapids.sh /home/maca-mcdf-3.7.0.3-linux-x86_64.tar.xz
# 装 rmmx/mcpy/numbax/mcdf 四个 cp310 wheel + 纯 Python 依赖
# (fastrlock / llvmlite==0.39.1 / pandas==1.5.3 / pyarrow==10.0.1 / protobuf==4.21.12)
# 并写入 numba->numbax 垫片
```

> ⚠️ **`/opt/conda` 的 `cudf` 即 mcdf(`23.02.00+b3.7.0.37`,面向 C500/MACA)。** C500 的 routing/REST 用它;**A100 的 routing/REST 用原生 NVIDIA RAPIDS cudf,装入独立 venv**(见 §5.2,避免覆盖 `/opt/conda` 的 mcdf)。numpy-only 的 LP/MILP/QP Python 路径两平台均可,无需 cudf。

### 0.6 数据集

`datasets/`(`RAPIDS_DATASET_ROOT_DIR`)。重新获取:`bash datasets/get_test_data.sh`(从 S3,需配置 `CUOPT_S3_URI` 与 AWS 凭据)。

---

## 1. C500 构建(`cpp/build_maca`)

```bash
cd <repo-root>

# 1) 清空构建目录
rm -rf cpp/build_maca && mkdir -p cpp/build_maca

# 2) MACA 环境(导出 MACA_PATH、cu-bridge PATH、MACA_DIRECT_DISPATCH=1、
#    以及 TBB_INCLUDE_DIR/TBB_LIBRARY/Boost_DIR/BZIP2_ROOT/数据集路径)
source maca_env.sh

# 3) 配置:cmake_maca 在 configure 期用真实 nvcc 探测,build 期切到 MACA 工具链
cmake_maca -S cpp -B cpp/build_maca -G Ninja \
  -DCUOPT_MACA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DBUILD_TESTS=ON \
  -DSKIP_GRPC_BUILD=ON \
  -DTBB_INCLUDE_DIR="$TBB_INCLUDE_DIR" \
  -DTBB_LIBRARY="$TBB_LIBRARY" \
  -DBoost_DIR="$Boost_DIR"

# 4) 构建(库 + CLI + 全部测试 + 示例)
ninja_maca -C cpp/build_maca -j "$(nproc)"
```

**产物**:`cpp/build_maca/libcuopt.so`(约 155 MB)、`cpp/build_maca/cuopt_cli`、35 个测试二进制(`cpp/build_maca/tests/<组>/`)。

**注**
1. `CMAKE_CUDA_ARCHITECTURES=80`:cu-bridge 经 `CUCC_TARGETS=xcore1000` 映射到 C500,沿用 sm_80 入口。
2. `FindTBB.cmake` / `find_package(Boost)` 只认 **CMake cache 变量**,不读同名环境变量;故 `TBB_*` 与 `Boost_DIR` 必须以 `-D` 传入(即便 `maca_env.sh` 已 export)。
3. 内存紧张时降低 `-j`(`PARALLEL_LEVEL=1 ninja_maca -C cpp/build_maca -j1` 单线程);本机内存充裕,`-j $(nproc)` 即可。

---

## 2. A100 构建(`cpp/build_cuda`)

A100 用标准 NVIDIA 工具链。**勿 `source maca_env.sh`**(它会把 cu-bridge 注入 PATH)。

```bash
cd <repo-root>
TC="$PWD/.toolchain"

# 1) 清空
rm -rf cpp/build_cuda && mkdir -p cpp/build_cuda

# 2) 环境:本地 CMake 3.30.8 + conda 的 BZip2
export PATH="$TC/cmake-3.30.8-linux-x86_64/bin:$PATH"
export BZIP2_ROOT=/opt/conda

# 3) 配置(CUOPT_MACA 默认 OFF)
cmake -S cpp -B cpp/build_cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DBUILD_TESTS=ON \
  -DSKIP_GRPC_BUILD=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTBB_INCLUDE_DIR="$TC/oneapi-tbb-2021.13.0/include" \
  -DTBB_LIBRARY="$TC/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8/libtbb.so" \
  -DBoost_DIR="$TC/boost-install/lib/cmake/Boost-1.84.0" \
  -DBZIP2_ROOT=/opt/conda \
  -DRAPIDS_DATASET_ROOT_DIR="$PWD/datasets" \
  -DFETCHCONTENT_SOURCE_DIR_CCCL="$TC/cccl-3.4.0-src"

# 4) 构建
ninja -C cpp/build_cuda -j "$(nproc)"
```

**产物**:`cpp/build_cuda/libcuopt.so`(约 150 MB)、`cuopt_cli`、35 个测试二进制。

**注**
1. `BZIP2_ROOT=/opt/conda`:系统无 `libbz2` 开发库,libbz2 在 conda 内。
2. **CCCL 必须用本地源**:`get_cccl.cmake` 固定 `GIT_TAG v3.4.0`,但公网 NVIDIA/cccl 仅有 `v3.4.0-rc0/-rc1/.dev` 标签、**无 `v3.4.0` 正式标签**,网络 `FetchContent` 必然 `Failed to checkout tag: 'v3.4.0'`。须用 `-DFETCHCONTENT_SOURCE_DIR_CCCL=` 指向本地 CCCL 3.4.0 源(本仓库已备 `.toolchain/cccl-3.4.0-src`)。
3. `TBB_*` / `Boost_DIR` 同 §1 注 2,必须 `-D`。
4. C500 build 用 MACA 自带 CCCL(`get_maca_cccl.cmake`,无 fetch),不受注 2 影响。

---

## 3. 运行 C++ 测试(功能矩阵第 1 节)

测试二进制位于 `cpp/build_<maca|cuda>/tests/<组>/<NAME>`,直接运行即可(顶层 `ctest` 未联动子目录,直接调用二进制最清晰)。每个二进制为一个 gtest 套件。

**运行环境**

```bash
# C500:
source maca_env.sh
export PATH="$PWD/cpp/build_maca:$PATH"          # CLI_TEST 会以子进程调用 cuopt_cli

# A100:
TC="$PWD/.toolchain"
export LD_LIBRARY_PATH="$PWD/cpp/build_cuda:$TC/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="$PWD/cpp/build_cuda:$PATH"           # 同上,cuopt_cli 须在 PATH
```

> **关键**:`CLI_TEST` 须能在 `PATH` 找到 `cuopt_cli`,否则 `wrong_parameter_type` / `partial_solution_file` 等用例失败。

**第 1 节对应的二进制**(`B` = `cpp/build_maca/tests` 或 `cpp/build_cuda/tests`):

| 功能面 | 二进制(相对 `B`) |
|---|---|
| LP — PDLP | `linear_programming/PDLP_TEST` |
| LP — 对偶单纯形 | `dual_simplex/DUAL_SIMPLEX_TEST` |
| LP — 单元 / Barrier | `linear_programming/LP_UNIT_TEST`、`linear_programming/MPS_PARSER_TEST` |
| MILP | `mip/MIP_TEST`、`mip/MIP_TERMINATION_STATUS_TEST`、`mip/PRESOLVE_TEST`、`mip/INCUMBENT_CALLBACK_TEST` |
| QP / SOCP | `qp/QP_UNIT_TEST`、`socp/SOCP_TEST` |
| C API | `linear_programming/C_API_TEST` |
| cuopt_cli | `utilities/CLI_TEST` |
| routing C++ | `routing/ROUTING_UNIT_TEST` |
| 距离引擎 | `distance_engine/WAYPOINT_MATRIXTEST` |
| examples | `examples/cvrp_daily_deliveries`、`examples/pdptw_mixed_fleet`、`examples/service_team_routing` |

**逐个运行示例**

```bash
B=cpp/build_maca/tests        # 或 cpp/build_cuda/tests
"$B/qp/QP_UNIT_TEST"
"$B/linear_programming/LP_UNIT_TEST"
"$B/routing/ROUTING_UNIT_TEST"
# ……
```

**一次跑完第 1 节的全部套件**

```bash
B=cpp/build_maca/tests        # A100 改为 cpp/build_cuda/tests
for t in \
  linear_programming/LP_UNIT_TEST linear_programming/PDLP_TEST \
  linear_programming/MPS_PARSER_TEST linear_programming/C_API_TEST \
  dual_simplex/DUAL_SIMPLEX_TEST qp/QP_UNIT_TEST socp/SOCP_TEST \
  mip/MIP_TEST mip/MIP_TERMINATION_STATUS_TEST mip/PRESOLVE_TEST mip/INCUMBENT_CALLBACK_TEST \
  routing/ROUTING_UNIT_TEST distance_engine/WAYPOINT_MATRIXTEST utilities/CLI_TEST ; do
  echo "== $t =="; "$B/$t" --gtest_brief=1 || echo "RC=$?"
done
```

示例(examples)在主机端校验数据后求解,直接运行:

```bash
"$B/examples/cvrp_daily_deliveries"   # 等
```

> **C_API_TEST 说明**:含若干 60s 墙钟时限用例,整体较慢;其中 128 线程时限可复现性用例在 C500 上偶发 1 例不复现(两次解均可行,4/8 线程可复现),属高并行时限项、非功能回归,详见 `maca_c500_results.md` §5。

---

## 4. Python:RAPIDS-free LP / MILP / QP / SOCP

`build_lp_modules_rapids_free.sh` 针对已构建的 `libcuopt.so` 编译 8 个 Cython 扩展(LP + routing + distance),`.so` 落在源码树。其中 **LP/MILP/QP 路径为 numpy-only,无需 RAPIDS**(两平台均可);routing/distance 模块运行时另需 mcdf(§5,仅 C500)。

```bash
# C500:经 cu-bridge 编译,使 .pyx 里的 cudaMemcpy(D2H)在编译期改写为 MACA 运行时
#       (链接 libruntime_cu/libmcruntime,而非 NVIDIA libcudart;否则解返回 inf)
source maca_env.sh
export PATH=/opt/conda/bin:$PATH                 # conda python3.10 优先
CUOPT_RT_MACA=1 LIBCUOPT_DIR=$PWD/cpp/build_maca \
  bash python/cuopt/build_lp_modules_rapids_free.sh

# A100:默认 NVIDIA 路径(g++ + libcudart)
export PATH="$PWD/.toolchain/cmake-3.30.8-linux-x86_64/bin:/opt/conda/bin:$PATH"
CCCL=$PWD/.toolchain/cccl-3.4.0-src LIBCUOPT_DIR=$PWD/cpp/build_cuda \
  bash python/cuopt/build_lp_modules_rapids_free.sh
```

运行测试:

```bash
PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
  python -m pytest python/cuopt/cuopt/tests/linear_programming -q
```

> **注**:`.so` 生成在源码树(`python/cuopt/cuopt/...`),C500 与 A100 版本相互覆盖;切换平台测试前须用对应平台重建这些模块。

---

## 5. Python:routing + distance

routing/distance 的 Python 路径需要 cudf/rmm/cupy/numba。**C500 用 MetaX mcdf 栈;A100 用原生 NVIDIA RAPIDS(装入独立 venv)。**

### 5.1 C500(mcdf)

C500 用 MetaX **mcdf** 栈(cudf/cupy 的 MACA 实现)。一键安装:

```bash
source maca_env.sh
export PATH=/opt/conda/bin:$PATH
bash python/cuopt/setup_maca_rapids.sh /home/maca-mcdf-3.7.0.3-linux-x86_64.tar.xz
# 安装 rmmx/mcpy/numbax/mcdf 四个 cp310 wheel(--no-deps)+ 纯 Python 依赖,
# 并写入 numba->numbax 垫片(cuOpt 编译进 .so 的 `from numba import cuda`)
```

运行测试(`CUDA_PATH` 指向 cu-bridge,供 cupy 在线 JIT 取 MACA 的 `cuda_fp16.h`):

```bash
source maca_env.sh
export PATH=/opt/conda/bin:$PATH
export CUDA_PATH=/opt/maca/tools/cu-bridge
PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
  python -m pytest python/cuopt/cuopt/tests/routing -q
```

### 5.2 A100(原生 RAPIDS,隔离 venv)

A100 用真正的 NVIDIA RAPIDS cudf。**装入独立 venv**(不继承 `/opt/conda` 的 site-packages),以免覆盖 C500 的 mcdf:

```bash
# 1) 建隔离 venv(python 3.10,与 cp310 模块 ABI 一致)
/opt/conda/bin/python -m venv .toolchain/a100-rapids-venv
VENV=$PWD/.toolchain/a100-rapids-venv
# 2) 装 RAPIDS(C 扩展依赖在 NVIDIA 源;cp310 取 24.12)
$VENV/bin/pip install --extra-index-url https://pypi.nvidia.com \
  "cudf-cu12==24.12.*" cupy-cuda12x numba cython pytest numpy
$VENV/bin/pip install pyyaml scipy networkx     # cuopt 纯 Python 依赖
$VENV/bin/pip install fastapi uvicorn msgpack    # REST 需要(§6)
```

> cuopt 的 pyproject 钉 `cudf-cu12==26.6.*`(随 26.06,公网未发布);routing 仅用 `cudf.DataFrame/Series/concat/from_pandas` 等稳定 API,24.12 即可。该 venv 为 numpy 2.x,须在 venv 内重建模块(numpy ABI)。

构建模块并测试(全程用 venv 的 python):

```bash
TC=$PWD/.toolchain
export PATH="$TC/cmake-3.30.8-linux-x86_64/bin:$VENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
# 关键:numba 默认会抓到 CUDA toolkit 的 stub libcuda(报 CUDA_ERROR_STUB_LIBRARY),
#       须显式指向真实驱动:
export NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so.1
CCCL=$TC/cccl-3.4.0-src LIBCUOPT_DIR=$PWD/cpp/build_cuda \
  bash python/cuopt/build_lp_modules_rapids_free.sh
export LD_LIBRARY_PATH=$PWD/cpp/build_cuda:$TC/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8:/usr/local/cuda/lib64
PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
  python -m pytest python/cuopt/cuopt/tests/routing -q
```

---

## 6. REST server(自托管)

**启动(C500)**:

```bash
source maca_env.sh
export PATH=/opt/conda/bin:$PATH
export CUDA_PATH=/opt/maca/tools/cu-bridge
export PYTHONPATH=$PWD/python/cuopt:$PWD/python/cuopt_server
python -m cuopt_server.cuopt_service &        # 默认 :5000
```

**启动(A100)**:用 §5.2 的 venv(原生 RAPIDS)+ 真实驱动:

```bash
VENV=$PWD/.toolchain/a100-rapids-venv; TC=$PWD/.toolchain
export PATH="$VENV/bin:$PATH" CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so.1
export LD_LIBRARY_PATH=$PWD/cpp/build_cuda:$TC/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8:/usr/local/cuda/lib64
export PYTHONPATH=$PWD/python/cuopt:$PWD/python/cuopt_server
export RAPIDS_DATASET_ROOT_DIR=$PWD/datasets
python -m cuopt_server.cuopt_service &        # 默认 :5000
```

冒烟(本机经 HTTP 代理,打本地端口须绕过代理):

```bash
# 1) 健康检查 -> 200
curl --noproxy '*' -s http://127.0.0.1:5000/cuopt/health
# 2) 异步提交 routing 请求 -> {"reqId":"..."}
REQ=$(curl --noproxy '*' -s -X POST http://127.0.0.1:5000/cuopt/request \
  -H "Content-Type: application/json" \
  -d @datasets/cuopt_service_data/cuopt_problem_data.json)
ID=$(echo "$REQ" | sed -nE 's/.*"reqId":"([^"]+)".*/\1/p')
# 3) 轮询结果(用 msgpack:默认 JSON 对含 ndarray 的结果会序列化报错,与求解无关)
curl --noproxy '*' -s -H "Accept: application/msgpack" \
  http://127.0.0.1:5000/cuopt/request/$ID -o /tmp/sol.msgpack
```

实测:**C500** health 200,首次提交触发内核重编译后 `solve success / 10.2s`;**A100**(venv)health 200,`solve success / 10.2s`。结果均经 msgpack 取回。

> solver worker 在 `process_async_solve` 中以 `SIGCHLD=SIG_DFL`(而非 `SIG_IGN`)运行,确保 C500 运行期内核重编译的 `mxcc` 子进程 `wait()` 可用;此修复已在源码中,无需手动设置。该重编译/`SIGCHLD` 问题仅 C500;A100 server 用原生 RAPIDS 无此问题。

---

## 7. 性能基准(LP / QP / routing)

完整基准见 `maca_c500_results.md` §3(Mittelmann LP / Maros–Meszaros QP / routing,`--time-limit 600` 等)。下列为命令与小规模对标方式。

**LP / QP(cuopt_cli,`--method 3` = barrier)** — 两平台通用,仅二进制路径与环境不同:

```bash
# C500: source maca_env.sh;  A100: 设置 §3 的 LD_LIBRARY_PATH
CLI=cpp/build_maca/cuopt_cli            # A100 改 cpp/build_cuda/cuopt_cli
"$CLI" --method 3 --time-limit 600 datasets/linear_programming/<instance>.mps
"$CLI" --method 3 --time-limit 600 datasets/benchmarks/maros_meszaros/<instance>.QPS
```

对标方法:同一 `.mps`/`.QPS` 在两平台运行,核对**目标值一致**、barrier 迭代数相近、耗时同量级。

**routing(`docs/dev/route_bench.py`,C500;依赖 mcdf)**:

```bash
source maca_env.sh && export PATH=/opt/conda/bin:$PATH
export CUDA_PATH=/opt/maca/tools/cu-bridge
PYTHONPATH=python/cuopt python docs/dev/route_bench.py <name> cvrp datasets/cvrp/Vrp-Set-X/X/X-n106-k14.vrp <n_veh> <budget_s>
# 输出 status / vehicles / cost / wall,核对解质量(cost)接近基线
```

> routing 基准用 cudf 构造输入,仅 C500;A100 routing 基准需原生 RAPIDS 环境(本机不具备,见 §0.5)。

---

## 8. 本次实测结果

### 8.1 C++ 功能矩阵(双平台 gtest)

构建:C500 `cpp/build_maca` / A100 `cpp/build_cuda`,均 `Release` + arch 80 + `BUILD_TESTS=ON`,全部成功。`--gtest_brief=1`,passed/total:

| 套件 | A100 | C500 | 备注 |
|---|---|---|---|
| QP_UNIT_TEST | 7/7 | 7/7 | |
| SOCP_TEST | 28/28 | 28/28 | |
| LP_UNIT_TEST | 35/35 | 35/35 | |
| MPS_PARSER_TEST | 138/138 | 138/138 | |
| DUAL_SIMPLEX_TEST | 30/30 | 30/30 | |
| PDLP_TEST | 70/70 | 70/70 | 各 +3 跳过 |
| MIP_TEST | 3/3 | 3/3 | |
| MIP_TERMINATION_STATUS_TEST | 12/12 | 12/12 | |
| PRESOLVE_TEST | 1/1 | 1/1 | |
| INCUMBENT_CALLBACK_TEST | 3/3 | 3/3 | |
| ROUTING_UNIT_TEST | 57/57 | 57/57 | |
| WAYPOINT_MATRIXTEST | 6/6 | 6/6 | |
| CLI_TEST | 7/7 | 7/7 | 需 cuopt_cli 在 PATH |
| C_API_TEST | 58/61 | 58/61 | 各 3 跳过(server/时限),0 fail;`maca_c500_results.md` §5 的 C500 128 线程可复现性项本次未触发 |

> A100 各套件耗时约为 C500 的 1/10(如 QP_UNIT:A100 0.45s vs C500 4.3s)。

### 8.2 Python 与 REST

| 项 | A100 | C500 | 命令 |
|---|---|---|---|
| LP/MILP/QP `pytest tests/linear_programming` | 51 passed / 9 skip / 0 fail | 51 passed / 9 skip / 0 fail | §4 |
| routing+distance `pytest tests/routing` | 43 passed / 0 fail | 42 passed / 1 fail | §5 |
| REST server(routing 端到端) | health 200;solve success / 10.2s | health 200;solve success / 10.2s | §6 |

> A100 的 routing/REST 依赖 cudf:经独立 venv 装原生 NVIDIA RAPIDS(§5.2,cudf 24.12)后**两平台均通过**;C500 用 mcdf。C500 routing 的 1 例失败为 numpy 告警顺序(非功能,§5);A100 的 cudf 24.12 不触发该告警,故 43/43。LP/MILP/QP 为 numpy-only,两平台均可。

### 8.3 性能基准(代表性对标)

| 实例 | 指标 | A100 | C500 |
|---|---|---|---|
| afiro LP(barrier) | 目标值 / 迭代 / 时间 | -4.64753135e+02 / 12 / 0.066s | -4.64753135e+02 / 12 / 0.088s |
| QBRANDY QP(barrier) | 状态 / 迭代 / 时间 | Optimal / 19 / 0.185s | Optimal / 19 / 0.384s |
| CVRP routing(route_bench, X-n106-k14, 15s) | cost / 车数 / wall | N/A | 26492 / 14 / 15.3s |

> 通过判据:两平台目标值一致、barrier 迭代相近,耗时同量级即可(A100 更快)。完整 600s 基准见 `maca_c500_results.md` §3。

---

## 9. 构建排错速查

| 现象 | 原因 / 处理 |
|---|---|
| `Could NOT find TBB` | `-DTBB_INCLUDE_DIR=`/`-DTBB_LIBRARY=` 必须显式传入(env 不被 `find_package` 读取) |
| `Could NOT find BZip2` | `export BZIP2_ROOT=/opt/conda`(A100) |
| `Failed to checkout tag: 'v3.4.0'`(A100) | 公网无 `v3.4.0` 正式标签;用 `-DFETCHCONTENT_SOURCE_DIR_CCCL=.toolchain/cccl-3.4.0-src` |
| `namespace "thrust" has no member "fill"`(A100) | NVIDIA CCCL 3.4 头文件更严格;已补 `#include <thrust/fill.h>` |
| `CLI_TEST` 部分用例失败 | `cuopt_cli` 未在 `PATH`;`export PATH=$PWD/cpp/build_<maca|cuda>:$PATH` |
| Python `get_vars()` 返回 inf(C500) | 模块未经 cu-bridge 编译;须 `CUOPT_RT_MACA=1` 重建(§4) |
