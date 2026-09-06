# Milestone 1B：Paged KV Cache 向量化写入与线程配置消融

日期：2026-09-06。InfiniLM 基线 `db25c65c81da760e7f4ac7d2b010eb08920fd006`，InfiniCore 基线 `012ea7e21deb38768b3c17a6f8e9c05e26566c5c`。GPU 为 RTX 4090 / SM89，CUDA 12.8，驱动与 NVML 均为 570.124.06。

## 实现

原 kernel 保持二维 `grid.x=head、grid.y=token`，每个 CTA 负责一个 token/head。原标量路径按 `threadIdx.x + n*NUM_THREADS` 遍历元素。本轮在相同 grid 和工作分配下增加显式 `INFINIOP_PAGED_CACHING_VECTOR=1` 开关：当 `INFINIOP_PAGED_CACHING_THREADS=32` 且条件满足时，32 个线程各处理 4 个 16-bit 元素，使用一个 8-byte `uint2` load 和 store；循环步长为 `32*4`，覆盖任意 4 对齐 head width。搬运只复制原始位模式。

`PagedCachingInfo` 现在保存 source/cache 最后一维 stride，标量路径也按该 stride 访问，保持非连续布局语义。向量路径要求：FP16/BF16、K/V head width 均为 4 的倍数、四个最后一维 stride 均为 1、token/head/block/slot 外层 stride 对应字节偏移均为 8 的倍数、K/V 实际 data pointer（包含 storage offset）均为 8-byte 对齐。任一条件不满足即回退到同一 32-thread 标量 kernel；FP32、非整齐维度、非连续最后一维和非对齐 storage offset 均走 fallback。负 slot 在两条路径都直接跳过。

隔离构建脚本 `build/caching.py` 增加 `--base-prefix`，最终链接一致复用已验证的 phase02 attention objects、device-link 和 archive，再替换 paged-caching object，避免旧 build 目录 object 混入。生产默认仍为 1024 threads，向量路径只通过实验开关启用。

## 实际生成证据

最终候选库：`/data/InfiniTensor/phase1b-cache-vector3/lib/libinfiniop.so`，SHA256 `5c86c826f54c9f0e7877c6ee19f39f3d4191257fe0741e2bd7193814046965b2`。正式 A/B/C/D 数据使用同一实现的 vector2 产物（SHA256 `7163e031d0c1a26c1a587682086eb88443efd5793ef848b7e67b6e4c6cbb03be`）；vector3 只增加关闭时无影响的实际 dispatch 诊断，并用于最终 correctness/memcheck。构建 manifest 位于 `runs/build_phase1b_vector3_20260906/build.json`，记录 InfiniCore SHA、base prefix、源码 hash、完整 nvcc/link 命令和输入。

运行时诊断（仅设置 `INFINIOP_PAGED_CACHING_DEBUG=1` 时输出）确认实际选择：

| 配置 | 实际 block | vector_requested | vector_eligible | vector_active |
|---|---:|---:|---:|---:|
| A 1024 标量 | 1024 | 0 | 1 | 0 |
| B 128 标量 | 128 | 0 | 1 | 0 |
| C 32 标量 | 32 | 0 | 1 | 0 |
| D 32 向量 | 32 | 1 | 1 | 1 |
| storage offset / stride / FP32 fallback | 32 | 1 | 0 | 0 |

诊断日志位于 `runs/m1b_diag_{A_default,B_128,C_32scalar,D_32vector}_20260906/` 和 `runs/m1b_vector_correctness_v3_20260906/`。`nm -C` 确认 32-thread scalar/vector、128、256、1024 模板均存在；对 `pagedCaching<half,32,true>` 的 SM89 object 用 `cuobjdump` 看到多处 `LDG.E.64` / `STG.E.64`。因此向量路径实际生成了 64-bit global load/store，未仅凭 C++ 类型推断。

## 正确性

`m1b_vector_correctness_v3_20260906` 覆盖 6 类 case，每类 eager 和 CUDA Graph：

- FP16/BF16、Dk=Dv=128、对齐 vector path；
- storage offset=1 fallback；
- innermost stride=2 fallback；
- Dk=130、Dv=96 fallback；
- FP32 fallback；
- page boundary slots `[0,255,256,511]`、负 slot `-1`，并检查 untouched region。

结果 `12/12 PASS`，每个 case 的完整 cache bitwise 比较、Graph replay 和负 slot 保留均通过。三份 `m1b_provider_{128,256,1024}_20260906` 各 16 行，合计正向 `36/36 PASS`，`skip_append` `6/6` 检出，`corrupt_written` `6/6` 检出。向量/fallback 综合 memcheck `ERROR SUMMARY: 0 errors`，记录于 `/tmp/m1b-vector-memcheck.log`。现有 InfiniCore `test/infiniop/paged_caching.py --nvidia` 全部支持 case 也通过（含 FP16/BF16/FP32、D=64/128、Dk=576/Dv=512、多 head）。

## A/B/C/D 消融

正式统计来自 16 个完整 run（两轮交替顺序 × A/B/C/D × FP16/BF16），每个 run 7 个 caching token case（1、4、16、32、255、256、257）和一个 B=4、history=256 的 one-token append→attention case；每个 case 20 个 CUDA-event 样本，caching workload 每样本 100 次，decode workload 每样本一次 append + 一次 attention。以下为两轮配对中位数，单位 μs，格式为 `eager / Graph`：

### FP16

| tokens | A 1024 标量 | B 128 标量 | C 32 标量 | D 32 向量 |
|---:|---:|---:|---:|---:|
| 1 | 12.746 / 1.516 | 12.844 / 1.413 | 12.953 / 2.212 | 12.930 / 1.423 |
| 4 | 12.708 / 1.505 | 12.859 / 1.403 | 12.943 / 2.166 | 12.869 / 1.413 |
| 16 | 12.723 / 1.649 | 12.810 / 1.556 | 12.910 / 2.335 | 12.862 / 1.564 |
| 32 | 12.718 / 2.427 | 12.851 / 1.618 | 12.969 / 2.509 | 12.882 / 1.761 |
| 255 | 14.566 / 13.865 | 12.882 / 2.888 | 12.913 / 3.308 | 12.872 / 2.611 |
| 256 | 14.085 / 13.406 | 12.836 / 2.898 | 12.948 / 3.297 | 12.895 / 2.616 |
| 257 | 14.100 / 13.432 | 12.830 / 2.918 | 12.936 / 3.369 | 12.853 / 2.627 |

### BF16

| tokens | A 1024 标量 | B 128 标量 | C 32 标量 | D 32 向量 |
|---:|---:|---:|---:|---:|
| 1 | 12.518 / 1.516 | 12.828 / 1.418 | 13.558 / 2.217 | 12.796 / 1.433 |
| 4 | 12.560 / 1.504 | 12.795 / 1.405 | 13.540 / 2.176 | 12.867 / 1.413 |
| 16 | 12.580 / 1.649 | 12.839 / 1.556 | 13.486 / 2.345 | 12.792 / 1.567 |
| 32 | 12.547 / 2.426 | 12.756 / 1.618 | 13.430 / 2.514 | 12.800 / 1.761 |
| 255 | 14.572 / 13.870 | 12.765 / 2.888 | 13.522 / 3.320 | 12.813 / 2.611 |
| 256 | 14.070 / 12.869 | 12.800 / 2.888 | 13.499 / 3.307 | 12.751 / 2.621 |
| 257 | 13.532 / 12.897 | 12.800 / 2.913 | 13.585 / 3.379 | 12.829 / 2.627 |

### append→attention

| dtype | A 1024 标量 | B 128 标量 | C 32 标量 | D 32 向量 |
|---|---:|---:|---:|---:|
| FP16 eager | 39.424 | 40.216 | 39.944 | 40.376 |
| FP16 Graph | 17.408 | 18.432 | 19.200 | 18.432 |
| BF16 eager | 38.400 | 40.192 | 40.976 | 39.904 |
| BF16 Graph | 16.880 | 18.432 | 19.168 | 17.920 |

A→B 的主要变化来自 CTA 线程数；B→C 在 32-thread 标量下变慢；C→D 在相同线程数下明显降低 Graph amortized 时间，并在长 token case 略降 eager 时间。向量路径没有稳定改善 one-token append→attention：FP16 Graph 为 18.432 μs，仍高于 A 的 17.408 μs；BF16 Graph 也只回到 17.920 μs，不能把 caching kernel 的收益等同于两算子收益。计时使用 CUDA events，eager 包含可能的 CPU 提交间隙；Graph 表示 replay 工作量。没有显存事务证据，不声称 HBM 带宽。

## 阶段判断

- **Implementation correctness：PASS**。完整 cache、untouched region、page boundary、负 slot、Graph 和 fallback 均通过。
- **Vectorization actually generated：PASS**。运行时 `vector_active=1`，SM89 SASS 出现 `LDG.E.64/STG.E.64`。
- **Kernel performance：SUPPORTED_GAIN（限定在已测 FP16/BF16、D=128、对齐布局）**。D 相对 C 的 Graph 约降低 20–36%，eager 约降低 0.5–1.0%；相对 A 的收益取决于 token count，不能跨 case 汇总。
- **Integration behavior：NO_CLEAR_GAIN**。append→attention 没有稳定收益，保留上一轮观察到的 Graph 回退信号。
- **Adoption：有限采用建议**。保留显式 vector 开关，适用于对齐的 FP16/BF16、Dk/Dv 为 4 倍数且 innermost stride=1 的 caching standalone 场景；不修改默认 1024，不推广到未验证 shape，不进入完整模型优化矩阵。

## 分仓库改动与建议提交信息

InfiniCore：`src/infiniop/ops/paged_caching/cuda/kernel.cuh` 增加 8-byte vector/scalar fallback 和完整最后一维 stride；`src/infiniop/ops/paged_caching/info.h` 保存新增 stride；`src/infiniop/ops/paged_caching/nvidia/paged_caching_nvidia.cu` 增加 32-thread/vector dispatch、对齐判定和诊断输出。

InfiniLM：`test/bench/decode_ops/build/caching.py` 增加 validated base-prefix 链接；`benchmarks/caching.py` 增加 32-thread/vector 参数；`common/timing.py` 统一真实 workload 执行次数；新增 `correctness/vectorized.py`、A/B/C/D raw runs、构建和诊断证据；本报告和 README 更新。未修改 attention 源码、默认 dispatch、公开 API 或驱动环境。

建议提交信息：

- InfiniCore：`perf(paged-caching): add aligned 32-thread vectorized writes`
- InfiniLM：`bench(decode): measure paged-cache vectorization and thread ablation`
