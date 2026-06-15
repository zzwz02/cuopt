# warp64（AMD MI100/CDNA）兼容性改造

> HIP 移植准备:消除对 warp size = 32 的硬编码假设,使全部 warp 级算法以
> **编译期常量**参数化(NVIDIA 32 / AMD wave64 64),零运行时开销。CUDA 上
> 已验证零回归;wave64 行为待 HIP 工具链可用后实测。

## 1. 中央抽象(全部 constexpr)

`cpp/vendor/include/raft/util/warp_constants.hpp`(新增,host-safe):

| 符号 | 含义 |
|---|---|
| `raft::WarpSize` | 32;HIP/CDNA 下经 `__AMDGCN_WAVEFRONT_SIZE__` 为 64 |
| `raft::WarpSizeLog2` | 移位用 log2(WarpSize) |
| `raft::lane_mask_t` | 每 lane 一位的掩码类型:32 位 / wave64 下 64 位 |
| `raft::LANE_MASK_ALL` | 全 lane 掩码(取代硬编码 0xffffffff) |

`cuda_dev_essentials.cuh` 新增设备助手(`if constexpr` 编译期派发):
`lane_popc` / `lane_ffs` / `lane_fls`(掩码位运算,自动选 `__popc`/`__popcll`
等 64 位变体)、`activemask()`(HIP 下为 `__ballot(1)`)、`laneId()` HIP 分支。
cuopt 侧 `routing/utilities/constants.hpp` 的 `warp_size` 改为
`raft::WarpSize` 别名。

## 2. 修复类别与位点

穷尽审计 204 个 warp 相关位点(4 类并行扫描):88 处既有 `raft::WarpSize`
用法在常量翻转后即正确(含全部 blockReduce/blockRankedReduce 的 shared
边界、`block_random_sample`、`block_inclusive_scan`);其余修复如下:

| 类别 | 位点 | 修复 |
|---|---|---|
| 掩码参数/返回值为 32 位 | raft warp_primitives(9 处)、cudart_utils `warp_full_mask()`、cuopt/eject 两套 shfl 包装 | `lane_mask_t` / `LANE_MASK_ALL` |
| ballot 结果存 32 位变量 + `__popc` | raft `binaryBlockReduce`、GES `binary_block_reduce`、`compute_fragment_ejections`(含 shared 掩码数组) | `lane_mask_t` + `lane_popc` |
| `__match_any_sync`/`__activemask` 掩码 | feasibility_jump `load_balancing.cuh` | 同上 + `lane_ffs` |
| 硬编码 lane 位运算 | load_balanced presolve:`threadIdx.x & 31`(3 处)、`31 - __clz(m)`、`1 << (5 - seg)` | `& (WarpSize-1)`、`lane_fls`、`WarpSizeLog2` |
| “单 warp”语义的块大小 32 | eject_until_feasible、vehicle_assignment、finalize_calc/upd 启动(4 处)+ graph node dim、sub-warp 调度 block_dim(4 处)、lane 数乘子(3 处) | `raft::WarpSize` |
| host 侧 warp 数取整 `(31+x)/32` | load_balanced_problem.cu | `(WarpSize-1+x)/WarpSize` |
| `cub::WarpReduce<T>` 默认逻辑宽度 | load_balanced presolve(2 处) | 显式 `<T, raft::WarpSize>` |
| wave64 下零长 shared 数组的实例化 | `kernel_get_best_insertion_ejection_solution<32,…>`(PDP/VRP) | 删除(仅 64 被实际启动);两处加 `static_assert(BLOCK_SIZE % WarpSize == 0)` |
| raft `logicalWarpReduceVector` 断言上限 32 | reduction.cuh | 放宽至 `<= WarpSize` |

**顺带修复 4 处既有断言括号缺陷**:`__popc(__activemask() == 1)` 实为对布尔
取 popc,应为 `__popc(__activemask()) == 1`(ejection_pool ×3、
execute_insertion ×1;route.cuh 5 处原本正确,统一迁移到
`lane_popc(activemask())`)。

## 3. 设计取舍

- 所有 warp 相关量保持 `constexpr`:数组边界、循环次数、移位、掩码在两个
  平台均编译期折叠,无运行时分支。
- 块尺寸桶(32/64/128/256 的 per-block 变体)中的 32 是普通块大小,非
  warp 语义,保留不动。
- 本次仅解决 **warp 大小**假设。CUDA 专属 API 的 HIP 映射(`__shfl_*_sync`
  的 `_sync`/mask 形态、`__syncwarp`、cooperative groups)属于 HIP 移植
  本体,届时在已收口的包装点(warp_primitives/cuopt_utils 两套包装)做
  平台分支即可。
- load_balanced presolve 的 1/2/4/…/threads-per-item 分桶方案与 warp 大小
  解耦(host 端公式已用 `raft::WarpSize`,device 端经
  `WarpSizeLog2 - seg` 自适应):wave64 下每 warp 自动承载 2× 项。

## 4. 验证

- CUDA(A100):全量重编零错误;全量 ctest **137/137**;afiro barrier
  目标值与基线**逐位一致**(-4.64753135e+02),即改造零数值影响。
- wave64 正确性经静态推演验证(掩码宽度、数组边界、归约树、bin 公式),
  待 MI100/HIP 工具链就绪后实测。
