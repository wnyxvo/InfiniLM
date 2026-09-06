# Milestone 1A.1：实验目录整理、测量协议修复与 CTA 候选重评

日期：2026-09-06。基线 InfiniLM `a518f2a6ab6519b125e57be109d2e867163a013e`、InfiniCore `012ea7e21deb38768b3c17a6f8e9c05e26566c5c`。默认 paged_caching 仍为 1024 threads；128/256 仅由显式环境变量选择。

## A：目录整理

已建立 `test/bench/decode_ops/` 稳定入口：

- `common/`：native 绑定、独立 FP32 reference、判定、计时和 provenance；
- `correctness/`：attention、caching、provider/Graph 入口；
- `benchmarks/`：修复后的 caching 与 append→attention benchmark；
- `build/`：隔离构建脚本；
- `reports/`：阶段报告和复现命令；
- `runs/`：本轮新增原始 JSON、日志、脚本快照和 patch。

`phase0/` 历史入口、历史 run 和原始 hash 保持不变；`phase0/correctness_support.py` 是兼容 wrapper，公共实现只有 `decode_ops/common/correctness_support.py`。新入口的 `--help`、import、CPU 计数测试均通过。

初始补测发现旧 `phase1a-cache-128` 仅替换了 caching object，最终共享库仍显式链接旧 attention object，provider 出现 NaN。已修复 `build/caching.py`，增加 `--base-prefix`，从已验证的 phase02 prefix 一致复用 attention objects、device-link 和 archive，再替换 caching object；没有覆盖旧产物。修复后的候选库为 `/data/InfiniTensor/phase1a-cache-128-fixed`。

## B：协议与构建证据

`common/timing.py` 让 workload、调用次数、归一化分母和公式由同一逻辑控制。caching workload 是一次 `paged_caching`，decode workload 是一次 append + 一次 attention；decode 不再额外除以 2 或 100。每条样本保存 `elapsed_ms`、`workload_calls`、算子组成、公式和最终 `normalized_us`。CPU callback 计数测试为 `calls=1 -> 1`、`calls=100 -> 100`，6 个 benchmark run 全部 PASS。

候选库构建 manifest 记录 InfiniCore SHA、base prefix、源码 hash、构建命令和产物 hash：

- 库：`/data/InfiniTensor/phase1a-cache-128-fixed/lib/libinfiniop.so`
- SHA256：`1f4542571e42c726945da1a2e4a137d0aca93bea24403dd336cba8824f6f5cef`
- `nm -C` 确认 FP16/BF16/FP32 的 `pagedCaching<128>`、`pagedCaching<256>`、`pagedCaching<1024>` 模板均存在。
- 正式计时关闭 dispatch/debug 输出；provider 日志另行保留实际 attention dispatch。

## 正确性、Graph 与负对照

驱动已恢复并在补测期间保持一致：`nvidia-smi` 和内核模块均为 `570.124.06`，PyTorch `2.9.1+cu128` 报告 `is_available=True`、RTX 4090；最小 CUDA 张量运算成功。

三份 provider run（`recheck_provider_fixed_{128,256,1024}_20260906`）各 16 行：12 行正向（FP16/BF16、三步增长、eager 与 InfiniCore framework Graph）和 4 行负对照。合计结果：

- 正向：`36/36 PASS`，整块 cache bitwise、untouched region 和独立 FP32 attention reference 均通过；
- `skip_append`：`6/6` 检出；`corrupt_written`：`6/6` 检出；负对照单独统计，不进入性能排名；
- 三个 caching 候选的 dispatch 均由相同 attention 配置复核，未复用旧候选负对照结果。

六份 benchmark run（两轮交替顺序、FP16/BF16、128/256/1024）共 48 行，全部 `target_correctness=PASS`、`cache_correctness=PASS`、`graph_correctness=PASS`。Graph 在捕获后预热，恢复状态后独立校验输出。

## 性能重测

每个候选/ dtype 运行 7 个 standalone caching token case（1、4、16、32、255、256、257）和一个 B=4、history=256 的真实 one-token append→attention case；每个 case 20 个样本，caching workload 每样本 100 次，decode workload 每样本 1 次 append + 1 次 attention。下表为两轮跨 token case 的中位数，单位 μs：

| dtype | caching CTA | caching eager | caching Graph | append→attention eager | append→attention Graph |
|---|---:|---:|---:|---:|---:|
| FP16 | 128 | 12.53 | 1.54 | 39.88 | 18.43 |
| FP16 | 256 | 12.72 | 1.56 | 40.17 | 18.43 |
| FP16 | 1024 | 12.94 | 2.45 | 39.46 | 17.41 |
| BF16 | 128 | 13.05 | 1.54 | 40.45 | 18.43 |
| BF16 | 256 | 12.78 | 1.56 | 39.98 | 18.41 |
| BF16 | 1024 | 13.63 | 2.45 | 39.67 | 17.34 |

相对 1024，standalone caching 的 128/256 在本矩阵中均有稳定下降（FP16 eager 约 1.7–3.2%，BF16 eager 约 4.3–6.3%；Graph 约 36%）；append→attention 端到端差异在轮次波动内，没有稳定收益。没有显存事务证据，因此不声称 HBM 带宽。

结论：standalone caching 候选收益为 **SUPPORTED_GAIN**；两算子 decode 结论为 **NO_CLEAR_GAIN**。候选不改变生产默认值，默认仍为 1024；在进入小规模 InfiniLM 模型验证前，应继续使用显式 128/256 实验开关并保持相同 attention 产物口径。

## Sanitizer 与范围

本轮 standalone caching memcheck 历史记录仍为 exit 0、`ERROR SUMMARY: 0 errors`。Phase 0.2 attention memcheck/racecheck 的历史 exit 139 继续单独标记 BLOCKED；NCU 的 `ERR_NVGPUCTRPERM` 仍不影响 CUDA-event 结果。本轮未扩大到完整模型 E2E、驱动安装或其他优化。

## 可复现命令

```bash
cd /data/InfiniTensor/InfiniLM
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/common/test_summarize.py
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/caching.py \
  --core /data/InfiniTensor/InfiniCore \
  --base-prefix /data/InfiniTensor/phase02-fixed-sm89 \
  --prefix /data/InfiniTensor/phase1a-cache-128-fixed \
  --evidence test/bench/decode_ops/runs/build_phase1a_cache_fixed_20260906
for n in 128 256 1024; do
  [ "$n" = 1024 ] && arg=default || arg=$n
  INFINI_ROOT=/data/InfiniTensor/phase1a-cache-128-fixed \
  LD_LIBRARY_PATH=/data/InfiniTensor/phase1a-cache-128-fixed/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/caching.py \
  --threads "$arg" --dtype fp16 --samples 20 --workload-calls 100 \
  --output test/bench/decode_ops/runs/recheck_bench_fp16_${n}_20260906
done
```

完整 provider 命令和逐样本结果位于 `test/bench/decode_ops/runs/recheck_provider_fixed_*_20260906/`；两轮 benchmark 位于 `recheck_bench_r{1,2}_*`。

建议提交信息：`refactor(bench): use validated attention base for CTA candidate rebuilds`。
