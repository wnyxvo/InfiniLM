# Milestone 1C — grouped CTA paged KV cache

本轮基线与实际 HEAD 一致：InfiniLM `2c03a958fedcdcfad7d2e065203cbd70299e3e73`，InfiniCore `15273ad835d279eb2f6952f83f79539b064bdcc6`。GPU 为 RTX 4090 (sm89)，CUDA 12.8，产物 `/data/InfiniTensor/phase1c-cache-grouped`，构建摘要保存在 `runs/build_phase1c_grouped_20260906/`，产物 fingerprint 为 `96b2e9097e85718d24d67aec9644a9536314cb3d5fd9198ffa000d7f7d657822`。

## 实现

`paged_caching` kernel 增加 `HEADS_PER_CTA` 模板参数和 `num_kv_heads` 尾部保护。单 head 候选保持原线程映射；E 使用 64 threads/CTA、2 warps，F 使用 128 threads/CTA、4 warps。`blockIdx.y` 是 token，`blockIdx.x` 是 head group；grouped 路径中 `warp=threadIdx.x/32`、`lane=threadIdx.x%32`，head=`blockIdx.x*HEADS_PER_CTA+warp`。每个 warp 只写自己的 head，维度循环为 `lane*4 + n*128`，对应 8-byte FP16/BF16 搬运。`ceil(Hkv/HEADS_PER_CTA)` 覆盖尾组，越界 warp 直接返回，因此不存在重叠写入。

保留现有 innermost stride、外层 8-byte stride 和指针对齐检查；不满足条件时 E/F 选择同样的 grouped scalar kernel。未修改 cache layout、公开 API、attention、dispatch 默认选择，也未加入跨 token 合并、shared memory 或 fusion。

## 计时协议修正

`measure_cuda` 负责循环 `workload_calls`，eager 回调只执行一次逻辑 workload。Graph 捕获一个 workload，`graph_workload_calls=1`，每次 replay 计一个 workload；原始 elapsed、replay/workload 数和 `elapsed_ms/workload_calls` 均写入 JSON。CPU 检查覆盖单次、100 次和已有批量对象（1 次 batch invocation 内含 100 个逻辑调用）。旧 raw 保留在历史 `m1b_*` 目录，不与本轮数据混表。

## 正确性

`runs/m1c_grouped_correctness_20260906/results.json`：10/10 PASS，包含 FP16/BF16 对齐向量、storage offset、innermost stride、Dk=130/Dv=96 fallback、FP32、E(Hkv=3/5) 和 F(Hkv=1/5)。每个 case eager/Graph bitwise cache 比较通过；负 slot 对应底层 block 未改变；Graph 在恢复原始 cache 后 replay 通过，并在同一图上更新 K/V 与合法 slots 后再次通过；跳过 replay 的负对照 10/10 被检测。`runs/m1c_memcheck_20260906` 的 grouped/fallback 代表矩阵 `compute-sanitizer memcheck` 为 `ERROR SUMMARY: 0 errors`。E/F smoke（含 B=4 append+attention）均 PASS。InfiniCore 原有 paged_caching 全回归在最终库上通过。

## 诊断与 SASS

E 诊断：`selected_threads=64 heads_per_cta=2 grid_x=4 num_heads=8 vector_active=1`；F 对应 128/4。`cuobjdump --list-text` 显示 `Li64ELb1ELi2E` 与 `Li128ELb1ELi4E` 实例；F FP16 实例包含 `LDG.E.64`/`STG.E.64`。CTA 数按 group 缩减，可能降低小 token/head 场景的并行度，故不预设 group 越大越快。

## B/D/E/F 消融（RTX 4090，FP16/BF16；每配置两轮交替，5 samples，eager 每 sample 20 workloads；Graph 每 replay 一个 workload）

下表为 token=1 的示例中位数（μs；完整逐样本数据在对应 run 目录，其他 token=4/16/32/255/256/257 同样保存）。

| dtype | cand | eager | Graph | append+attention eager | append+attention Graph |
|---|---:|---:|---:|---:|---:|
| FP16 | B 128 scalar | 13.52–14.79 | 14.34 | 45.06 | 18.43–19.46 |
| FP16 | D 32 vector | 13.52–13.62 | 14.34–15.30 | 43.01–46.08 | 19.39–20.35 |
| FP16 | E 64/2 vector | 13.72–15.97 | 14.34–15.39 | 44.03–46.11 | 19.46 |
| FP16 | F 128/4 vector | 13.98–14.94 | 14.34–15.36 | 44.03–46.08 | 19.46 |
| BF16 | B | 13.57–15.93 | 14.34 | 44.03 | 18.43 |
| BF16 | D | 13.67–13.93 | 14.34 | 43.97–44.03 | 18.43–19.46 |
| BF16 | E | 13.77–13.82 | 14.34–14.40 | 43.01–44.03 | 18.43 |
| BF16 | F | 13.98–14.90 | 14.34–16.38 | 44.03–47.10 | 18.43–19.46 |

E/F 没有相对 B/D 的稳定额外收益；部分轮次小 token 有轻微退化，符合 CTA 数减少后的并行度效应。append+attention 主要由 attention 占用，算子级差异被淹没；不能把历史 18.4 对 17.4 的差异直接归因于噪声，本轮已在相同 Graph 拓扑和明确 workload 计数下复测。固定热 cache 使用连续 slot；跨页、非连续 slot 和负 slot 在正确性矩阵覆盖。没有 HBM 事务数据，因此不声称带宽收益。

## 集成与建议

固定历史 256、每活跃序列追加一个 token 的 B=4 验证通过；B=4 Graph workload 明确为一次 append+attention。另以相同拓扑运行 B=16：B/D/E/F Graph 分别为 40.45/42.43/52.74/40.96 μs，均 PASS；E 的 grouped CTA 在该批量仍未优于单 head，说明减少 CTA 的收益不足以抵消并行度损失。按结果保留 E/F 为显式实验开关，不修改生产默认；小 batch 使用 B/D 单 head 规则更稳妥，只有在实际 head/token 组合显示稳定收益时才考虑 grouped。下一阶段若继续，应先做有界的 InfiniLM 集成 workload（含明确 batch Graph 计数），不启动整模型性能矩阵。

## 变更、命令与建议提交信息

InfiniCore：`src/infiniop/ops/paged_caching/cuda/kernel.cuh`、`src/infiniop/ops/paged_caching/nvidia/paged_caching_nvidia.cu`。InfiniLM：`benchmarks/caching.py`、`common/timing.py`、`correctness/vectorized.py`、本报告及 `reports/README.md`。构建：

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/caching.py --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase1c-cache-grouped --base-prefix /data/InfiniTensor/phase02-fixed-sm89 --evidence test/bench/decode_ops/runs/build_phase1c_grouped_20260906
```

正确性、memcheck、诊断和性能命令均由各 run 的 `manifest.json` 保存；性能命令模板为 `INFINI_ROOT=/data/InfiniTensor/phase1c-cache-grouped LD_LIBRARY_PATH=/data/InfiniTensor/phase1c-cache-grouped/lib:/root/.infini/lib:/usr/local/cuda/lib64 .../benchmarks/caching.py --threads {32|64|128} --grouped-heads {1|2|4} [--vector] --dtype {fp16|bf16} --samples 5 --workload-calls 20 --output <run>`。

建议 commit message：`feat(infiniop): add grouped CTA paged caching candidates`；`bench(decode_ops): fix workload accounting and add milestone 1C evidence`。

验证命令（完整路径）：

```bash
INFINI_ROOT=/data/InfiniTensor/phase1c-cache-grouped LD_LIBRARY_PATH=/data/InfiniTensor/phase1c-cache-grouped/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/vectorized.py --output test/bench/decode_ops/runs/m1c_grouped_correctness_20260906
INFINI_ROOT=/data/InfiniTensor/phase1c-cache-grouped LD_LIBRARY_PATH=/data/InfiniTensor/phase1c-cache-grouped/lib:/root/.infini/lib:/usr/local/cuda/lib64 compute-sanitizer --tool memcheck --error-exitcode 99 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/vectorized.py --output test/bench/decode_ops/runs/m1c_memcheck_20260906
INFINI_ROOT=/data/InfiniTensor/phase1c-cache-grouped LD_LIBRARY_PATH=/data/InfiniTensor/phase1c-cache-grouped/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python /data/InfiniTensor/InfiniCore/test/infiniop/paged_caching.py --nvidia
```
