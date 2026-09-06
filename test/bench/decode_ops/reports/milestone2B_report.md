# Milestone 2B — shared-KV GQA Split-KV decode

本里程碑在 InfiniCore `ee2ca9c383af0b203ffaa842c276288756f41e2d`、InfiniLM `1c35eb42d5bb01aa8e6a74b18ccf4d708e93e655` 上实现可切换的 HD128 GQA shared-KV Split-KV CUDA kernel，范围为 Hq=32、Hkv=8、ratio=4、FP16/BF16。每个 CTA 对应一个 KV head、sequence 和 shard；四个 warp 分别处理四个 Q head。每个 token tile（8×128）由 CTA 协作加载到 shared K/V，四个 warp 在同步区间内复用该 tile，并各自维护 FP32 `(m,l,acc)`；现有 FP32 workspace/combine 保持不变。空 shard 写入 `m=-inf,l=0,acc=0`，页表和跨页尾部按 token 映射，所有 shared-memory barrier 由完整 CTA 执行。

最终重建 prefix 为 `/data/InfiniTensor/phase2b-gqa-shared-fixed`，`lib/libinfiniop.so` SHA256 为 `7b7717ebaf96852a4edab49162535d968efab0710afd8fb1973ec2004904c303`。实现文件为 [kernel_v2.cuh](/data/InfiniTensor/InfiniCore/src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh) 与 [paged_attention_hd128.cu](/data/InfiniTensor/InfiniCore/src/infiniop/ops/paged_attention/nvidia/paged_attention_hd128.cu)。开关为 `INFINIOP_FLASH_GQA_SHARED_SPLITKV=1`，同时需要 Split-KV、CTA、无 ALiBi 且 Hq=4×Hkv；dispatch 记录为 `splitkv_gqa_shared`。默认行为未改变。

## 正确性

`runs/m2b_gqa_shared_correctness_20260906` 与最终重建复核 `runs/m2b_gqa_shared_final_correctness_20260906`、`runs/m2b_gqa_shared_final_splits_20260906`：42/42 cases PASS（2/4/8 splits 各 14 cases）（FP16/BF16；L=1/7/255/256/257/2049 与 `[7,256,2049]` 混合 batch；eager、CUDA Graph、Graph 后原地更新 Q/cache/lengths 均通过；skip-replay sentinel 负控 PASS）。FP16 最大误差约 0.0009，BF16 最大误差约 0.0073，均在既定阈值内。代表性 2/4/8 split 的 kernel 索引沿用同一 workspace 协议，三组 split 均实测通过。

## 完整 attention 性能

`runs/m2b_attention_perf_n100_20260906` 与 `runs/m2b_attention_perf_n500_20260906` 对每个 path 测量完整 attention（partial + combine），Graph 每图 N 次逻辑 workload、每样本 replay 2 次，并按 N×2 归一化；N=100/500 结果收敛。B=1 代表值（μs，Graph median）：

| path | L=256 | L=2049 |
|---|---:|---:|
| production_default（实际 splitkv_cta） | 8.79 | 46.45 |
| non_split_cta | 22.11 | 156.57 |
| gqa_fused | 53.69 | 390.15 |
| ordinary warp Split-KV | 27.36 | 180.78 |
| ordinary CTA Split-KV | 8.80 | 46.45 |
| 2A independent-K/V 4-warp | 27.54 | 182.93 |
| 2B shared-KV tile | 28.87 | 192.78 |

B=16,L=2049 的 2B shared-KV 为约 301.4 μs；该标量 shared-memory 原型相对 ordinary CTA Split-KV 为 `NO_GAIN`，但它验证了共享 tile、独立 softmax 与 combine 的正确端到端路径。没有 HBM 流量计数，因此不宣称 HBM 减少。

## Sanitizer 与集成

compute-sanitizer memcheck、racecheck、synccheck 代表矩阵均 PASS（0 errors / 0 hazards，证据目录分别为 `runs/m2b_gqa_shared_memcheck_20260906`、`runs/m2b_gqa_shared_racecheck_20260906`、`runs/m2b_gqa_shared_synccheck_20260906`）。KV Update standalone 通过。初始组合构建曾因 host link 只取 base archive 中旧的 `pagedCaching<T,512/1024/4096>`，缺少当前源码需要的 `pagedCaching<T,256,false,1>`，触发 `CUDA error: named symbol not found`；已修正 `build/attention.py`，将当前 cache object 显式加入最终 `.so`。修复后 `runs/m2b_integration_fixed_b4_20260906` 的 8 个 paged-caching workload 与 append+attention 均 PASS（target/cache/Graph 均 PASS）。整模型 E2E 为 `NOT_RUN`。

## 复现

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/attention.py --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase2b-gqa-shared --base-prefix /data/InfiniTensor/phase02-fixed-sm89 --cache-prefix /data/InfiniTensor/phase1c-cache-grouped --evidence test/bench/decode_ops/runs/build_phase2b_gqa_20260906
INFINI_ROOT=/data/InfiniTensor/phase2b-gqa-shared LD_LIBRARY_PATH=/data/InfiniTensor/phase2b-gqa-shared/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/gqa_shared.py --output test/bench/decode_ops/runs/m2b_gqa_shared_correctness_20260906
INFINI_ROOT=/data/InfiniTensor/phase2b-gqa-shared LD_LIBRARY_PATH=/data/InfiniTensor/phase2b-gqa-shared/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/attention.py --output test/bench/decode_ops/runs/m2b_attention_perf_n100_20260906 --samples 3 --graph-workloads 100 --graph-replays 2
```

建议提交信息：`feat(infiniop): add shared-kv gqa split-kv decode`；`bench(decode_ops): add milestone 2B correctness and performance evidence`。
