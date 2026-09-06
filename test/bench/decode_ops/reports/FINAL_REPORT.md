# FINAL_REPORT — CUDA LLM Decode 算子优化收束

日期：2026-09-06。基线为 InfiniLM `0bd5dbcc7fe39e57cf5671573a9729281d143be4`、InfiniCore `5e3d5b32209623d246cd2cbd01fbfc33d4b67b67`；两仓库均先核对为 clean。最终统一库 `/data/InfiniTensor/phase4-final/lib/libinfiniop.so`，SHA256 `fbabcb6dcc63245b1c14bc850a7f59e884b5024104d0cd56b3896cb6951813d5`，由当前 Core hd128 对象、已验证 attention 对象和 `phase1c-cache-grouped` caching 对象链接生成。

## 项目解决的问题与调用链

项目围绕 paged KV decode，补齐了 CUDA paged-caching 写入、paged-attention GQA/Split-KV 路径、独立 FP32 reference、append→attention 验证和 Graph 计时。真实验证链为 `infinicore.ops.paged_caching → infinicore.ops.paged_attention`；InfiniLM Llama 当前仍在模型层使用 dense `grouped_query_attention`，没有 paged provider 接口，因此整模型 E2E 为 `NOT_RUN`。

上游原有实现提供基础 paged cache/attention 与默认 dispatch。本项目新增或修正了 8-byte FP16/BF16 vector copy、CTA right-sizing/grouped CTA、GQA Split-KV/shared-KV/tile-softmax 实验、capacity_v1 dispatch 以及本轮 NUM_SPLITS 优先级和页容量边界修复。shared-KV/tile-softmax 保留实现与负结果，已 CLOSED。

## CUDA 修复与 dispatch

- `DECODE_SPLITKV=auto + NUM_SPLITS=<正整数>`：正整数优先，strategy 不覆盖。
- `NUM_SPLITS=auto` 或未设置：允许 heuristic/capacity_v1；非法值安全回退生产默认四分片。
- 显式 `DECODE_SPLITKV=0`：关闭 split，保持 GQA fused/default 语义。
- capacity_v1 仅限显式 auto、CTA、Hq/Hkv=32/8、FP16/BF16、无 ALiBi；范围外回退原 heuristic。
- capacity_v1 选 `num_splits=1` 时走普通 `cta_nosplit`，不再误入 GQA fused。
- provider 页表按所有 decode step 的 `history+4` 预留；`--history` 明确表示调用前历史长度。

`m4_strategy_override_20260906` 覆盖 numeric2、auto、unset、closed、invalid、out_of_scope 六类实际 dispatch；`m4_provider_history510_20260906` 通过页容量边界回归，均 PASS。

## KV Update 最终验收

候选 A=生产默认、B=128-thread scalar、D=32-thread 8-byte vector；FP16/BF16 各三轮交替，tokens=1/32/256，Graph 捕获 20 个逻辑调用、replay=2，分母为 `workloads_per_graph × graph_replays`。下表为三轮中位数，单位 µs/逻辑调用（eager / Graph）：

| dtype | tokens | A | B | D |
|---|---:|---:|---:|---:|
| FP16 | 1 | 14.213 / 1.638 | 12.984 / 1.510 | 13.271 / 1.523 |
| FP16 | 32 | 14.356 / 2.509 | 13.107 / 1.690 | 13.281 / 1.818 |
| FP16 | 256 | 14.888 / 13.824 | 13.107 / 2.970 | 13.179 / 2.700 |
| BF16 | 1 | 13.373 / 1.587 | 12.941 / 1.510 | 13.230 / 1.536 |
| BF16 | 32 | 13.394 / 2.483 | 13.190 / 1.690 | 13.578 / 1.818 |
| BF16 | 256 | 14.868 / 13.798 | 13.046 / 2.970 | 13.128 / 2.688 |

B 相对 A 在本矩阵中具有稳定收益：Graph 绝对差（B−A）为 FP16 `-0.128/-0.819/-10.829`、BF16 `-0.077/-0.793/-10.828` µs（tokens=1/32/256）；D−B 分别为 FP16 `+0.013/+0.128/-0.282`、BF16 `+0.026/+0.128/-0.282` µs。三轮 Graph 中位数波动最大为 A 在 tokens=32 的 FP16 `2.483–2.534` µs，其他候选均在约 `0.03` µs 内。D 相对 A 仅在部分 token 范围改善，且相对强对照 B 没有稳定优势，因此最终建议保留 B/D 为显式候选，生产默认仍为 A。FP16 rotating-slot D（tokens=32）为 `13.394 / 1.818`，slots 与输入均在计时前准备；该结果只作合法布局对照，不等同 HBM 带宽测试。

KV Update 最终采用口径的 18 个 A/B/D runs 全部 PASS；B 代表点 memcheck 为 `ERROR SUMMARY: 0 errors`。append→attention 在 B4/B16、FP16/BF16 的 cache bitwise、attention 输出、eager 和 Graph 均 PASS。单算子延迟改善不外推为整个 decode 加速。

## Attention、策略和集成结论

GQA Split-KV、shared-KV、tile-softmax 均已实现并完成 correctness/Graph/sanitizer；shared/tile 相对普通 CTA 强基线为 `NO_GAIN`。capacity_v1 相对原 auto 在 B4/L2049 修复错误的 GQA fused 选择，约恢复到生产默认水平；相对生产默认在 B1/L256、B16/L2049 有退化或持平，没有泛化为生产加速的依据。策略只保留为显式实验开关，不增加 capacity_v2 或继续搜索阈值。

验证层级如下：

| 分类 | 成果 |
|---|---|
| IMPLEMENTED | vector/scalar fallback、grouped CTA、GQA/shared/tile kernel、capacity_v1、NUM_SPLITS 优先级、history+4 页容量 |
| CORRECTNESS_VERIFIED | FP16/BF16 cache、attention、append、page relocation、Graph、负控；history=510；strategy override；代表性 memcheck |
| PERFORMANCE_GAIN_WITH_SCOPE | B 的 128-thread scalar 相对 A，在 tokens 1/32/256、FP16/BF16、当前 SM89 测试矩阵有效 |
| NO_GAIN | D 相对 B 无稳定优势；shared-KV/tile-softmax 相对普通 CTA；capacity_v1 相对生产默认无稳定净收益 |
| NOT_RUN/BLOCKED | InfiniLM 整模型 paged-attention E2E；未下载模型、未改 dense 模型架构 |

## 最小复现与交付

```bash
INFINI_ROOT=/data/InfiniTensor/phase4-final \
LD_LIBRARY_PATH=/data/InfiniTensor/phase4-final/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
python test/bench/decode_ops/correctness/strategy_override.py \
  --output test/bench/decode_ops/runs/m4_strategy_override_20260906

INFINI_ROOT=/data/InfiniTensor/phase4-final \
LD_LIBRARY_PATH=/data/InfiniTensor/phase4-final/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
python test/bench/decode_ops/correctness/provider.py \
  --variant strategy --history 510 \
  --output test/bench/decode_ops/runs/m4_provider_history510_20260906
```

构建证据在 `runs/build_phase4_final_20260906/`；KV Update 三轮证据为 `runs/m4_kv_{fp16,bf16}_{A,B,D}_r{1,2,3}_20260906/`；集成为 `runs/m4_integration_{fp16,bf16}_B{4,16}_20260906/`；sanitizer 为 `runs/m4_kv_B_memcheck_20260906/`。

推荐保留生产默认行为；显式采用 B 需由上层 workload 配置 `INFINIOP_PAGED_CACHING_THREADS=128`，D 仅用于已验证对齐布局实验。不要把 provider 级集成称作 InfiniLM 整模型集成。

建议提交信息：

- InfiniCore：`fix(paged-attention): preserve explicit split overrides and safe fallbacks`
- InfiniLM：`test(decode): finalize KV update and dispatch acceptance`
