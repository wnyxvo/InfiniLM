# Milestone 2C — GQA shared-KV tile-level online softmax

基线已核实为 InfiniLM `fdc3fea8274cab33431e3487d83d526d7b3c42b3`、InfiniCore `fbe837d7d3a82cd59c5b3117e4b211fb859e2f83`。本轮在 2B shared-KV kernel 上实现 tile-level online softmax，范围为 Hq=32、Hkv=8、D=128、FP16/BF16、无 ALiBi。CTA 映射保持为 sequence × KV head × shard，四个 warp 分别处理四个 Q head，tile=8、split=4 的默认实验配置不变。每个 warp 的 lane 持有 Q 的 4 个维度；lane 0 暂存 tile 内最多 8 个 score/prob，所有 lane 协作 QK reduction 和 V 累加。

每个有效 tile 先计算全部 token score，再由 lane 0 求 tile 最大值和 `m_new`，计算 `alpha=exp2(m_old-m_new)`、各 `p_j=exp2(score_j-m_new)` 与 tile 权重和；`l` 更新一次，历史 `acc` 缩放一次，然后累加本 tile 的加权 V。shared K/V 单缓冲的两处 `__syncthreads()` 保证加载、消费和覆写顺序；空 shard 不进入 tile 循环并写中性 partial。没有引入 cp.async、双缓冲、Tensor Core 或 cache layout 改动。

修改文件为 [kernel_v2.cuh](/data/InfiniTensor/InfiniCore/src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh)、[attention.py](/data/InfiniTensor/InfiniLM/test/bench/decode_ops/build/attention.py)、[gqa_shared.py](/data/InfiniTensor/InfiniLM/test/bench/decode_ops/correctness/gqa_shared.py) 与 [benchmarks/attention.py](/data/InfiniTensor/InfiniLM/test/bench/decode_ops/benchmarks/attention.py)。最终候选库 `/data/InfiniTensor/phase2c-gqa-tile` 的 `libinfiniop.so` SHA256 为 `5eb3bceb4bad6d683060ad4b2b611129ab408dd7aa6ebb1f55bc06eb780e88a3`。

## 正确性与 Graph

`runs/m2c_gqa_tile_correctness_formal_20260906`：60/60 PASS（FP16/BF16；2/4/8 splits 各 20 cases；L=1/7/8/9/255/256/257/2049/8192 与混合长度；跨页、非连续物理页、尾 tile、空 shard、全部 Q heads、batch=3）。Eager、Graph 以及 Graph 捕获后原地更新 Q/K/V/合法 lengths 均通过。skip-replay 负对照使用正式 `check_output` 返回 FAIL（nonfinite sentinel），再将该预期失败记为 PASS。

## 性能与假设

`runs/m2c_attention_fp16_r1_20260906`、`r2`、`r3` 为三轮交替顺序的完整 attention（含 combine），Graph N=100、replay=2、归一化分母 N×2；`runs/m2c_attention_bf16_r1_final_20260906` 为 BF16 代表轮。FP16 B=1,L=2049 的三轮中位数（μs）：production default 46.50，普通 CTA Split-KV 46.49，2B/当前 shared tile-softmax 约 210.92；B=16,L=2049 当前候选约 303.3。BF16 B=1,L=2049：production 46.71，CTA 43.20，tile-softmax 196.79。N=100/500 的 2B 历史结果用于归一化交叉检查；本轮未改变超参数。

结论：Tile 级 softmax `IMPLEMENTED`；相对 2B shared-KV `NO_GAIN`（仍为同一标量 shared-KV 映射，性能量级未改善）；相对普通 CTA Split-KV 强基线 `NO_GAIN`。瓶颈解释是基于源码和延迟的假设：tile 内仍由每个 warp 串行处理 8 个 token，且每 tile 保存 score/prob 标量并重复读取 V；未采集硬件计数器，不能把寄存器压力或 HBM 流量写成实测结论。按停止条件不再增加第二计算映射候选。

## Sanitizer 与集成

最终候选的 compute-sanitizer 代表矩阵均通过：`runs/m2c_gqa_tile_memcheck_20260906`、`runs/m2c_gqa_tile_racecheck_20260906`、`runs/m2c_gqa_tile_synccheck_retry_20260906`，分别为 0 errors、0 hazards、0 errors；synccheck 外层退出码为 0。两算子 native 集成 `runs/m2c_integration_b4_20260906` 在 B=4 长上下文（含 append+attention、Graph）全部 PASS。整模型 E2E 为 `NOT_RUN`。

## 复现命令

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/attention.py --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase2c-gqa-tile --base-prefix /data/InfiniTensor/phase02-fixed-sm89 --cache-prefix /data/InfiniTensor/phase2b-cache-current --evidence test/bench/decode_ops/runs/build_phase2c_gqa_tile_20260906
INFINI_ROOT=/data/InfiniTensor/phase2c-gqa-tile LD_LIBRARY_PATH=/data/InfiniTensor/phase2c-gqa-tile/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/gqa_shared.py --output test/bench/decode_ops/runs/m2c_gqa_tile_correctness_formal_20260906
INFINI_ROOT=/data/InfiniTensor/phase2c-gqa-tile LD_LIBRARY_PATH=/data/InfiniTensor/phase2c-gqa-tile/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/attention.py --output test/bench/decode_ops/runs/m2c_attention_fp16_r1_20260906 --samples 2 --graph-workloads 100 --graph-replays 2 --dtype fp16 --round 1
```

建议提交信息：`feat(infiniop): optimize gqa shared-kv tile softmax`；`bench(decode_ops): add milestone 2C evidence`。
