# Milestone 1A：Paged KV Update CTA Right-sizing

日期：2026-09-06。基线 InfiniLM `2085ff4733fa5f5d17281e7af709fc72a622eb46`、InfiniCore `bd17d120b97de7bdf88a7b79a102e7088f656856`。GPU 为 RTX 4090，CUDA 12.8，sm_89。已有 GQA owner-thread 与 attention shared-buffer 同步修复保持不变。

## 结论

**Adoption decision：NO_CLEAR_GAIN，保持默认 1024 threads。**

128-thread 候选在单独 paged_caching event 测量中通常比 1024-thread 默认低约 1–3%（eager），Graph replay 约低 10% 左右；256-thread 没有稳定优于 128。两轮独立进程、FP16/BF16、token count 1/4/16/32 均通过 bitwise cache correctness。但真实 append→attention 两算子路径在三个配置间约 0.38–0.40 μs/调用，差异落在测量波动内，不能据此宣称端到端收益。没有把候选接入默认选择逻辑。

## 工作映射与唯一变量

`paged_caching_nvidia.cu` 实际 launch 为 `grid=(num_kv_heads,num_tokens,1)`：x 是 KV head，y 是 token，每个 CTA 只处理一个 token/head。`kernel.cuh` 中 K/V 分别执行 `for (i=threadIdx.x; i<head_size/v_head_size; i+=NUM_THREADS)` 标量复制；padding slot `<0` 直接返回。D=128 时 1024 threads 中只有前 128 个线程各处理一个元素，其余线程无该循环迭代；这描述的是工作分配，不是 occupancy 或带宽推断。

本轮唯一变量是 `NUM_THREADS`：默认 1024，显式环境变量 `INFINIOP_PAGED_CACHING_THREADS=128|256` 选择已有模板实例化。grid、slot 语义、stride、cache layout、dtype、拷贝宽度和 attention 路径不变。默认未设置环境变量时仍走 1024。

源码注释已更正为 2D token/head grid 和 scalar strided copy，避免与实现不符的“1D/vectorized”描述。

## 产物和 provenance

候选库由 `build_caching_isolated.py` 从 bd17 当前源码、记录的 XMake recipe、CUDA 12.8、sm_89 构建，位于持久目录 `/data/InfiniTensor/phase1a-cache-128/lib/libinfiniop.so`，SHA256 `cba73e0e23dd046211da23799be51d4678e01cd41bde6bd37b4e36586b263e4b`。该库同时包含 128、256 和默认 1024 模板；仅通过环境变量改变选择。旧 `/root/.infini/lib/libinfiniop.so` 未覆盖。

## Correctness 与 integration

`caching_benchmark.py` 对 FP16/BF16、token count 1/4/16/32 做整 cache bitwise 比较，包含 untouched region；append→attention 使用真实 C API 的 paged_caching 后接 PagedAttention，并用独立 FP32 reference 检查。所有 6 个候选/类型组合均 PASS。现有 InfiniCore `test/infiniop/paged_caching.py --nvidia` 在候选库、默认 1024 选择下 PASS，覆盖 FP32、不同 Dk/Dv、非连续 view 和多 head 配置；native caching smoke 的 128/256 候选也 PASS。skip/corrupt 负对照沿用 Phase 0.2 provider 证据，未计入性能样本。

## 性能证据

每个点为独立进程、10 个 CUDA-event 样本，每个样本执行 100 次后除以 100；两轮顺序分别为 default→128→256 和 256→default→128。GPU 空闲，关闭 dispatch/debug 输出。下表为代表性中位数（μs/单次调用，round1/round2）：

| dtype | tokens | 1024 eager | 128 eager | 256 eager | 1024 Graph | 128 Graph | 256 Graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 1 | 12.979/12.964 | 12.831/12.733 | 12.974/12.667 | 1.505/1.505 | 1.326/1.331 | 1.341/1.341 |
| FP16 | 16 | 12.861/12.791 | 12.733/12.642 | 12.693/12.497 | 1.669/1.669 | 1.485/1.485 | 1.495/1.495 |
| FP16 | 32 | 12.882/12.856 | 12.667/12.601 | 12.611/12.605 | 2.447/2.437 | 1.536/1.546 | 1.556/1.556 |
| BF16 | 1 | 13.000/12.912 | 12.744/12.580 | 12.790/12.662 | 1.505/1.505 | 1.331/1.321 | 1.341/1.341 |
| BF16 | 16 | 12.852/12.749 | 12.703/12.538 | 12.652/12.365 | 1.669/1.669 | 1.485/1.485 | 1.495/1.495 |
| BF16 | 32 | 12.861/12.764 | 12.652/12.529 | 12.759/12.426 | 2.437/2.447 | 1.536/1.536 | 1.556/1.556 |

append→attention（B=4，lengths `[1,4,16,32]`，53 tokens）中位数：FP16 default/128/256 为 0.399/0.389/0.399 μs（round1），round2 为 0.390/0.389/0.384；BF16 为 0.394/0.395/0.389，round2 为 0.399/0.384/0.383。该路径没有稳定收益，且该数值包含两次 C API 调用的批量 event 口径，不能外推整模型 E2E。

完整 raw/manifest 位于 `test/bench/phase0/phase1a_runs/`，大 trace、模型和共享库之外的临时文件不归档。负对照和失败结果不进入性能统计。

## Sanitizer 状态

本轮 standalone caching memcheck（128 threads、FP16、小用例）exit 0，`ERROR SUMMARY: 0 errors`，记录于 `/tmp/m1a-memcheck.log` 和 `phase1a_runs/memcheck_caching_128_20260906`。Phase 0.2 attention memcheck/racecheck 的 exit 139 仍是历史 BLOCKED，未改写为本轮通过。NCU 仍受 `ERR_NVGPUCTRPERM` 限制，不阻塞 CUDA-event 实验。

## Adoption 与后续边界

不修改默认 caching 选择逻辑；128 候选仅保留为显式实验开关。理由是 isolated caching 有趋势，但 append→attention 未显示超过波动的稳定收益，尚不足以改变生产默认行为。整模型 E2E、KV update vectorization、append fusion、attention 优化和性能计时均不在本轮范围。

分别判定：Correctness **PASS（覆盖范围内）**；Performance evidence **PARTIAL/NO_CLEAR_GAIN**；Integration **PASS（append→attention 参考校验）**；Sanitizer **PASS（caching memcheck）/BLOCKED（历史 attention mem/race）**；Adoption **REJECTED for default change**。

建议提交信息：

- InfiniCore：`experiment(paged-caching): expose CTA thread candidates`
- InfiniLM：`bench(milestone1a): measure paged caching right-sizing`

简历事实表述：完成 paged KV cache CTA 线程数的 CUDA-event A/B、Graph 和 append→attention 正确性实验；观察到候选趋势但未形成可确认端到端收益，因此保留默认实现。不要写成已获得稳定加速。
