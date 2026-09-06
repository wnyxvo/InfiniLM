# Milestone 2A — GQA-aware Split-KV attention prototype

基线 HEAD 与任务要求一致：InfiniLM `e03f3c42276f793d4be9a1c252ce8a02593fddae`，InfiniCore `840eabd0215c0c6b3d21d394d374e63f560a31ab`。GPU RTX 4090 / sm89，CUDA 12.8。

## KV Update 收尾

沿用 Milestone 1C 的 A/B/D/E/F 结果；本轮没有新增 group size。批量 Graph 计时协议已在 `benchmarks/caching.py` 固化：Graph 内捕获一个逻辑 workload，`measure_cuda` 管理 replay；B/D/E/F 两轮、FP16/BF16 和 B=4/B=16 集成均通过。结论按 shape 为 `NO_BENEFIT`（E/F 相对 B/D 无稳定收益，小 token 偶有退化），保留显式实验开关，生产默认不变。

## 原型与映射

普通 Split-KV 原本是一个 CTA/Q-head/shard，GQA 只在每个 CTA 内计算 `kv_head = q_head / 4`，因此同一 KV head 会被 4 个 CTA 重复读取。新增 `flashAttentionDecodeSplitKvWarpKernel<...,4>`：grid.x 为 `ceil(Hq/4)=Hkv`，grid.y 为 sequence，grid.z 为 split；CTA 128 threads 的 4 个 warp 分别计算共享 KV head 的 4 个 Q heads。每个 warp 独立维护 `(partial_m, partial_l, partial_acc)`，workspace 仍为 `[split, seq, Q-head, ...]`，combine 使用现有稳定 softmax 合并（空 shard 为 `m=-inf,l=0,acc=0`）。

该最小原型复用了现有数值路径和 combine，但尚未把 K/V tile 放入 shared memory 供四个 warp 复用；因此“减少 CTA/逻辑映射”已实现，真实 global-load 复用仍是待验证假设。未修改公开 API、cache layout 或默认 dispatch；设置 `INFINIOP_FLASH_GQA_SPLITKV=1`、`INFINIOP_FLASH_DECODE_SPLITKV=1`、`INFINIOP_FLASH_DECODE_KERNEL=cta` 才选择新路径。

## 正确性

`runs/m2a_gqa_correctness_20260906`：14/14 PASS（FP16/BF16，B=1/3，L=1/7/255/256/257/2049，混合 `[7,256,2049]`，4 splits，eager 和 CUDA Graph，全部 Q heads）。独立 FP32 reference 与 native 结果比较通过，Graph 使用同一 descriptor/输入指针。新路径 memcheck 运行记录于 `runs/m2a_gqa_memcheck_20260906`；compute-sanitizer 在当前环境发生进程 SIGSEGV（退出 139，无 device error 输出），因此 sanitizer 状态为 `BLOCKED`，不把该结果当作 PASS。旧 paged-caching memcheck 仍为 0 errors。

## 完整 attention 消融

`runs/m2a_attention_perf_20260906` 使用相同输入和包含 combine 的完整调用；Graph 每图 100 次逻辑 workload、每样本 replay 2 次，归一化分母为 200，eager 每样本 10 次调用。代表性中位数（μs）：

| path | B=1,L=256 | B=1,L=2049 | B=16,L=2049 |
|---|---:|---:|---:|
| default/auto baseline | 22.1 Graph | 169.3 Graph | 215.2 Graph |
| GQA fused | 53.7 | 389.2 | 392.7 |
| ordinary Split-KV + combine | 8.8 | 46.5 | 214.4 |
| GQA-aware Split-KV prototype | 27.6 | 197.7 | 336.7 |

所有配置 correctness PASS。原型因没有 shared K/V tile 复用，当前相对普通 Split-KV 为 `NO_GAIN`，短序列和大 batch 还有明显退化；不继续搜索更多 split/group size。workspace 为 descriptor 预留的 8-split FP32 partial 区域，实际 `num_splits=4`。

## InfiniLM 集成

使用现有 KV Update + append→attention 两算子路径完成 B=4 长上下文代表点和 B=16 退化对照，attention 候选只通过环境开关切换；cache append 与 attention reference 均 PASS。整模型 E2E：`NOT_RUN`（本轮没有用两算子结果替代整模型吞吐）。

## 状态与建议

- KV Update 收尾：`DECIDED / NO_BENEFIT`。
- 新 attention kernel：`IMPLEMENTED`（数值与普通路径一致；shared-load 复用仍是假设）。
- 正确性：`PASS`；新路径 sanitizer：`BLOCKED`（环境 SIGSEGV）。
- 性能：`NO_GAIN`。
- 两算子集成：`PASS`。
- 整模型 E2E：`NOT_RUN`。

可写入简历的实际贡献是：在现有 paged Split-KV/combine 上实现可切换的 GQA-aware 4-warp CTA 映射、workspace 索引和空 shard 中性处理，并完成独立 reference、Graph 和性能证据。shared-memory K/V 复用、HBM 流量和整模型加速仍是实验假设，不能写成已验证收益。

## 文件、构建与复现

InfiniCore 修改：`src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh`、`src/infiniop/ops/paged_attention/nvidia/paged_attention_hd128.cu`。InfiniLM 新增：`test/bench/decode_ops/build/attention.py`、`correctness/gqa_split.py`、`benchmarks/attention.py` 及本报告和索引。

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/attention.py --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase2a-gqa-split --base-prefix /data/InfiniTensor/phase02-fixed-sm89 --cache-prefix /data/InfiniTensor/phase1c-cache-grouped --evidence test/bench/decode_ops/runs/build_phase2a_gqa_20260906
INFINI_ROOT=/data/InfiniTensor/phase2a-gqa-split LD_LIBRARY_PATH=/data/InfiniTensor/phase2a-gqa-split/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/gqa_split.py --output test/bench/decode_ops/runs/m2a_gqa_correctness_20260906
INFINI_ROOT=/data/InfiniTensor/phase2a-gqa-split LD_LIBRARY_PATH=/data/InfiniTensor/phase2a-gqa-split/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/attention.py --output test/bench/decode_ops/runs/m2a_attention_perf_20260906 --samples 3 --graph-workloads 100 --graph-replays 2
```

建议提交信息：`feat(infiniop): prototype gqa-aware split-kv decode`；`bench(decode_ops): add milestone 2A attention evidence`。

## 后续校正（Milestone 2B 前置审计）

此前表格中的 `default/auto baseline` 实际来自旧脚本把 `INFINIOP_FLASH_DECODE_SPLITKV=0` 固定为 CTA non-split；它不是源码环境下的 production default。后续脚本已拆分为 `production_default` 与 `non_split_cta`，因此旧表不能用于宣称 production default 性能。此前 sanitizer 的 SIGSEGV 只证明该 compute-sanitizer 运行在当前机器上退出 139；没有足够证据把原因归因于环境，故应表述为 `BLOCKED / cause unresolved`。两算子集成的 PASS 仅覆盖实际运行过的 KV Update + attention workload，provider 与整模型 E2E 仍未验证。
