# Milestone 3A — CTA Split-KV 形状适配与 Provider 集成

已核实基线：InfiniLM `1e86f33ed05a9e90b47e420424d9aee31b0c7772`，InfiniCore `348b26c234d87ef507f697b19692ec6bdaa8ba4a`。InfiniCore 的现有高效路径为 CTA Split-KV，默认 threads=64、tile=8；合法固定 split 为 1/2/4/8，workspace 预留最多 8 splits，combine 始终在 split 路径执行。原 auto 使用 `max_num_blocks_per_seq × page_size` 作为 seqlen 上界，并按 SM 数、CTA waves、每 shard 工作量和 combine wave 估算 split。

## Shared-KV 收束

2B/2C 的 `gqa_splitkv_shared` 与 `gqa_splitkv_tile` 实际使用同一 shared-KV dispatch；报告已不再把它们作为两个独立消融。shared-KV 与 tile-softmax 均已实现，但没有超过普通 CTA 强基线，本轮关闭该分支，不再增加候选。历史缺失日志不补造；2B/2C sanitizer 和集成状态以各自实际 artifact 为准。

## 策略与实现

新增显式实验策略 `INFINIOP_FLASH_SPLITKV_STRATEGY=capacity_v1`，仅在 `INFINIOP_FLASH_DECODE_SPLITKV=auto` 时生效，显式 numeric split 仍优先。策略只使用调用链已有的 host metadata：`num_heads`、`num_seqs`、`max_num_blocks_per_seq`、`page_block_size` 和设备 SM 数，不引入 device→host 拷贝或同步。容量策略为：容量≤256 或 batch≥8 选 1；B≤2 时容量≥1024/4096 选 4/8，否则 2；B=3–7 时容量≥2048 选 4、≥1024 选 2，否则 1。策略适用 NVIDIA、HD128、Hq=32/Hkv=8、FP16/BF16、无 ALiBi；其它形状回退原逻辑。

策略在 Graph capture 时冻结；Graph replay 不重新运行 host dispatch，输入长度变化需要既有 bucket/重捕获机制。诊断输出增加 `strategy=capacity_v1`、实际 path 和 split。

## 有界消融

`runs/m3a_attention_fp16_r{1,2,3}_20260906` 使用三轮交替顺序、完整 attention（含 combine）、Graph N=100、replay=2。开发点为 B=1,L=256；B=1,L=2049；B=4,L=2049；B=16,L=2049，另有短/长容量对照。FP16 B=1,L=2049 三轮 Graph median（μs）：production 46.46，原 auto 42.99，capacity_v1 42.97，普通 CTA Split-KV 42.86。B=4,L=2049：原 auto 390.17，capacity_v1 56.56；B=16,L=2049：原 auto 393.19，capacity_v1 393.24。原 auto 在该形状使用容量上界导致不必要的 GQA fused/非 split 选择；capacity_v1 修正了 B=4 的明显不合适选择。B=1,L=2049 与 B=16 无稳定额外收益，因此结论为相对原 auto `GAIN`（限定 B=4 形状区域），相对固定强 CTA `NO_GAIN`，整体配置适配为 `GAIN_WITH_SCOPE` 而非新算法加速。

FP16 三轮均 PASS；BF16 代表点沿用同一 attention harness，结果 PASS。所有计时记录实际 dispatch 日志，错误配置未进入性能排名。未开展 threads/tile 扩展搜索。

## 留出验证与 Provider

未参与阈值选择的验证点为 provider 的 B=4、历史长度 255/256/257/8，且包含页表重定位和三步 append；并包含 B=4 中的短序列对照。真实 InfiniCore provider 入口 `infinicore.ops.paged_caching` + `infinicore.ops.paged_attention` 已运行：

- `runs/m3a_provider_strategy_final_20260906`：FP16/BF16，eager 与 InfiniCore Graph，正向 12 行及负对照 4 行全部 PASS；策略 trace 明确记录 `capacity_v1` 和实际 dispatch。
- `runs/m3a_provider_auto_20260906`、`runs/m3a_provider_default_20260906`：原 auto/default 对照均 PASS。
- `runs/m3a_provider_strategy_memcheck_20260906`：memcheck `ERROR SUMMARY: 0 errors`。

KV Update 使用调用前独立 expected cache，append、attention、Graph 和负对照均经过验证。整模型 E2E 为 `NOT_RUN`。

## 状态

- Shared-KV 分支：`CLOSED`
- 配置适配：`GAIN_WITH_SCOPE`（B=4 长上下文相对原 auto；相对固定 CTA 强基线 `NO_GAIN`）
- 相对原 auto：`GAIN`（有限形状）
- 正确性：`PASS`
- Graph：`PASS`
- Sanitizer：`PASS`（新增策略 memcheck）
- Provider：`PASS`
- 整模型：`NOT_RUN`

最终候选库 `/data/InfiniTensor/phase3a-strategy`，`lib/libinfiniop.so` SHA256 为 `dadb339dae38d266dd3c4958f490d75ad37957912d0b7e7adce1facc5a02d0f3`。

## 复现命令

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/build/attention.py --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase3a-strategy --base-prefix /data/InfiniTensor/phase02-fixed-sm89 --cache-prefix /data/InfiniTensor/phase2b-cache-current --evidence test/bench/decode_ops/runs/build_phase3a_strategy_20260906
INFINI_ROOT=/data/InfiniTensor/phase3a-strategy LD_LIBRARY_PATH=/data/InfiniTensor/phase3a-strategy/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/provider.py --variant strategy --output test/bench/decode_ops/runs/m3a_provider_strategy_final_20260906
INFINI_ROOT=/data/InfiniTensor/phase3a-strategy LD_LIBRARY_PATH=/data/InfiniTensor/phase3a-strategy/lib:/root/.infini/lib:/usr/local/cuda/lib64 /data/InfiniTensor/InfiniLM/test/bench/decode_ops/benchmarks/attention.py --output test/bench/decode_ops/runs/m3a_attention_fp16_r1_20260906 --dtype fp16 --round 1
```

建议提交信息：`feat(infiniop): add capacity-aware split-kv dispatch strategy`；`bench(decode_ops): add milestone 3A provider evidence`。

### 3B errata

3A 的 `capacity_v1` 在选择 `num_splits=1` 时实际会落入 GQA fused；3B 已将该选择改为显式 `cta_nosplit`。3A 基准的 auto/capacity kernel 计数曾按模式固定，3B 改为读取实际 dispatch；`gqa_splitkv_shared` 与 `gqa_splitkv_tile` 也不再作为两条独立收益路径。3A provider 只覆盖 history≤257，不能证明长序列收益；3B 新增 B4、L=2049 的 eager/Graph provider 验证。长序列 capacity 相对 auto 的收益来自路径选择，不能解读为超过生产默认。
