# Milestone 3B — Split strategy and kernel path decoupling

基线已核对：InfiniLM `39cbb9fc8a8ece581ca14f39491891b562af3865`、InfiniCore `38fd472f605dd4706f8bc23907dfc6788ff03b0a`，两树均无未提交改动（源码修改由本交付工作区记录）。目标形状为 Hq=32、Hkv=8、D=128，无 ALiBi，覆盖 FP16/BF16。

## 修正

`INFINIOP_FLASH_SPLITKV_STRATEGY=capacity_v1` 只在显式 `DECODE_SPLITKV=auto`、CTA、GQA(32/8)、FP16/BF16 且无 ALiBi 时生效；显式 numeric `NUM_SPLITS` 保持优先，范围外回退原 heuristic。策略选 `num_splits=1` 时现在明确进入 `cta_nosplit`，不会再误入 `cta_gqa_fused`。dispatch 日志同时记录 path、split、num_splits、CTA tile/threads、capacity strategy 标志。Graph 在 capture 时冻结该配置。

基准脚本在计时前独立捕获实际 dispatch，记录 workspace、实际 split、kernel launches/workload、capacity blocks、库 hash；Graph 使用 `N×replays` 逻辑 workload 归一化，完整 attention（含 combine）计时。此前 `gqa_splitkv_shared`/`gqa_splitkv_tile` 的重复别名已去掉后者。此前 3A 的 B4 长序列收益来自 auto 选择 GQA fused 与 capacity 选择 split 的路径差异，并非新 kernel 单独超过生产默认；此前 provider 只覆盖短历史，现补充 L=2049。

## 结果

6 个基准 run（FP16/BF16 各三轮，45 rows/run，全部 PASS）覆盖 B1 L256、B1/B4/B16 L2049，以及 B4 L2049 的 capacity=9/16 对照。代表性 Graph 中位数（µs/token-workload，三轮中位数）如下：

| dtype | shape | production default | original auto | capacity_v1 | ordinary non-split CTA |
|---|---|---:|---:|---:|---:|
| FP16 | B1 L256 | 9.06 | 54.02 | 22.48 | 22.43 |
| FP16 | B4 L2049 cap9 | 61.44 | 422.76 | 61.34 | 170.13 |
| FP16 | B16 L2049 cap9 | 215.19 | 425.32 | 232.45 | 232.29 |
| BF16 | B4 L2049 cap9 | 60.72 | 427.01 | 60.83 | 178.59 |

capacity=16 与 capacity=9 的 B4 路径和延迟保持一致（FP16 61.44/61.49 µs，BF16 60.88/60.83 µs），说明选择基于 host capacity 元数据而非 D2H/sync/autotune。强基线是生产默认：capacity_v1 在 B4 长序列约持平，短序列和 B16 低于默认；其可验证收益是避免 original auto 的错误 GQA fused 选择。

## 正确性、集成与 sanitizer

- 真实 `infinicore.ops.paged_caching + paged_attention` provider：B4、L=2049、pages=9，FP16/BF16 eager 与 InfiniCore Graph 均 PASS；默认/auto/strategy 三种配置均 PASS。strategy dispatch 为 `splitkv_cta,num_splits=4`，auto 为 `cta_gqa_fused,num_splits=1`；短历史 strategy 的 split=1 为 `cta_nosplit`。
- provider 的 append、page relocation、mixed lengths、Graph replay，以及 skip-append/corrupt-written 两个负控均 PASS（负控被检测）。
- strategy 长序列 `compute-sanitizer --tool memcheck`：0 errors。
- 这是 InfiniCore API/provider 级集成 PASS。当前 InfiniLM Llama 模型层仍使用 dense `grouped_query_attention`，没有 paged-attention provider 调用链，因此“整模型 InfiniLM E2E”在本里程碑标记 NOT RUN，不把 provider 结果冒充模型 E2E。

## 交付件

- 候选库：`/data/InfiniTensor/phase3b-strategy/lib/libinfiniop.so`
- SHA256：`5a540c4ada3502ec2945f9799ad5bf0badf6ae275cba3d6c091854dedb2573b0`
- 构建证据：`test/bench/decode_ops/runs/build_phase3b_strategy_20260906/`
- 长序列 provider：`test/bench/decode_ops/runs/m3b_provider_{strategy,auto,default}_long_20260906/`
- 性能：`test/bench/decode_ops/runs/m3b_attention_{fp16,bf16}_r{1,2,3}_20260906/`
- sanitizer：`test/bench/decode_ops/runs/m3b_provider_strategy_memcheck_20260906/`
