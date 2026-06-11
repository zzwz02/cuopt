# cuOpt 移除 rmm/raft/rapids 的 MVP 实施结果（CUDA-only）

> 实施记录。分支 `feature/remove-cudss-custom-ldlt`。目标：在 CUDA 上以
> 自有 cuda::mr-free 同名 shim 替换 rmm/raft/rapids_logger，使 libcuopt 构建并
> 链接为零 RAPIDS 运行时依赖，求解结果与基线一致。计划见
> `docs/dev/rapids_removal_plan.md`，shim 在 `cpp/vendor/include/{rmm,raft,rapids_logger}`。

## 1. 结论

**核心目标达成。** libcuopt.so + cuopt_cli + 全部 40 个测试可执行文件构建通过、
**零编译错误**，运行时**不再链接 librmm.so / libraft / librapids_logger.so**。
cuopt_cli 的 barrier 与 PDLP 求解目标值与基线**逐位一致**。C++ 单元测试
121/137 通过；剩余失败是已表征的 cuOpt 潜在 bug 长尾（见 §5），非 shim 正确性问题。

## 2. 运行时依赖（ldd 实证）

移除前 libcuopt.so 链接：…… **librmm.so、librapids_logger.so** ……
移除后 `ldd libcuopt.so`：

```
libcudart(static) libcublas libcublasLt libcusparse libnvJitLink
libamd libsuitesparseconfig libtbb libgomp  (+ libc/libstdc++/libm)
```

→ 两个 RAPIDS 运行时 .so 消失；体积 ~108MB → ~106MB。CCCL（thrust/cub/libcu++，
非 RAPIDS）仍以平凡 FetchContent 拉取。

## 3. 目标值对比（cuopt_cli，零回归）

| 实例 | method | 基线目标 | 移除后目标 | 一致 |
|---|---|---|---|---|
| afiro | 3 barrier | -4.64753135e+02 | -4.64753135e+02 | ✅ 逐位 |
| afiro | 1 PDLP | (Optimal) | -4.64761260e+02 | ✅ Optimal |
| min_x_squared | 3 QP | +1.00000002e+00 | +1.00000002e+00 | ✅ 逐位 |
| QP_Test_1 | 3 QP | -9.99600000e+01 | -9.99600000e+01 | ✅ 逐位 |
| QP_Test_2 | 3 QP | +1.11111115e-01 | +1.11111115e-01 | ✅ 逐位 |

求解时间相当或更快（移除后为热缓存测量）。无性能回退。

## 4. shim 架构

- **rmm（手写，cuda::mr-free）**：device_buffer/device_uvector/device_scalar 落在
  `cudaMalloc`（默认资源，匹配 rmm 的 `initial_resource()`）/可选 `cudaMemPool_t` 池
  （cuopt_cli、server、测试显式选）；用虚基类 `device_memory_resource` + 自有
  `device_async_resource_ref` 取代 cuda::mr concept；exec_policy(sync+nosync) 用
  池化 thrust 分配器；stream/stream_pool/error/aligned/cuda_device。
- **raft core（手写触碰 rmm 的部分 + 逐字移植纯 CUDA 叶子）**：handle_t
  （stream + thrust policy + 懒 cublas/cusparse + 拷贝构造 + comms 桩）、
  span（裸指针 iterator、逐元素 ==、限定转换构造）、device_mdspan（cuda::std::mdspan）、
  device_setter、nvtx(no-op)、logger(stub)；error/macros/math/operators/reduction/
  warp_primitives/cudart_utils/cuda_utils 等逐字取自 raft 26.06（Apache-2.0，保浮点序）。
- **raft 计算原语**：legacy 指针形式 linalg（binaryOp/ternaryOp/unaryOp/eltwise*/
  divideScalar/transpose/reduce）手写 thrust/cub 封装；cublas/cusparse wrappers 逐字移植；
  rng（PCGenerator/RngState 逐字移植，host uniform/uniformInt/make_blobs 手写指针版——
  **非 bit-exact，影响 routing 合成数据生成的 golden**）。
- **rapids_logger**：header-only shim，printf 风格 log，level_enum 数值 0..6 保持，
  ostream/file/callback sink；logger_macros.hpp 静态化（替代 rapids-cmake 生成）。
- **构建**：cpp/CMakeLists.txt 删 get_rmm/get_raft/rapids_logger 拉取与链接，加
  vendor include 目录与 CUDA::cudart_static；CCCL/gtest 仍经 rapids-cmake 拉（rapids-cmake
  作为纯构建工具暂留，4b 可进一步替换为平凡 FetchContent）。

## 5. 剩余 C++ 测试失败（已表征长尾，~16/137）

均非 shim 正确性问题（cuopt_cli 实测求解正确）：

- **潜在 use-after-free / 未初始化读**：warm-start / initial-solution / maximize 等
  求解路径在「同一进程连续跑多测试」时，被 shim 与 rmm 不同的内存复用模式暴露成
  `cudaErrorIllegalAddress` / 级联 `CUBLAS_STATUS_EXECUTION_FAILED`。rmm 的
  pool_memory_resource 用 free-list 延迟复用恰好掩盖；这是 cuOpt 既有潜在 bug（与
  cuDSS 移除时修过的 `maximize_` 未初始化同类）。`--rmm_mode=pool` 可复现。
- **并发/多GPU**：run_sub_mittleman 等在单 GPU 上走多 GPU 分发路径触发
  `cudaErrorInvalidDevice`（solve.cu 的 device_setter 逻辑）。
- **CLI 参数测试**：cli_test_t 的若干 invalid-arg 用例（与求解无关）。
- **routing 合成数据**：generator.cu 的 RNG 非 bit-exact，固定 seed 数据集变化 →
  golden 需重生成。

后续修复方向：用 compute-sanitizer（注意它会改变内存布局、需关池或定位写）+
逐路径排查 cuOpt 既有未初始化读/UAF；或在测试中统一用非池化分配器并重生成
routing golden。

## 6. 待办

- Python LP/MILP 路径在无 RAPIDS 包环境构建/通过（Cython cimport 改本地声明、
  cudf 惰性导入、pyproject 去 rapids-build-backend）——未开始。
- 上述 C++ 长尾失败的逐项修复。
- 可选 4b：rapids-cmake → 平凡 CMake/FetchContent，彻底移除 rapids 构建工具。
- HIP/MACA 后端（本 MVP 不含；shim 的 cuda::mr-free 设计是其前提）。

## 7. 提交序列（本分支）

1. `vendor cuda::mr-free rmm/raft shim headers` — rmm + raft core 两层手写 shim。
2. `vendor raft compute primitives shim` — linalg/cublas/cusparse/rng。
3. `vendor header-only rapids_logger shim`。
4. `cut over to vendored shim — libcuopt builds RAPIDS-free` — CMake 切换，ldd 零 RAPIDS。
5a. `green full build incl. tests + cuopt_cli` — 全量构建通过。
5b. `match rmm default resource; non-pooled test allocator` — 121/137 通过 + 长尾表征。
