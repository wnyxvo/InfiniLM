# decode_ops 实验入口

这里维护当前 decode correctness、paged caching 和 CUDA-event 实验。`test/bench/phase0/` 保留为历史入口与原始归档；新实验统一写入 `test/bench/decode_ops/runs/`。共享 native 绑定、独立 FP32 reference、判定和 provenance 位于 `common/`。

当前状态：Milestone 1A.1 已完成，Milestone 1B 的 32-thread、8-byte FP16/BF16 vector path 及 scalar fallback 已通过 correctness、Graph 和 memcheck。A/B/C/D 消融表明向量路径改善 standalone caching，append→attention 没有稳定收益；默认 caching 仍为 1024。Phase 0.2 attention memcheck/racecheck 的历史 exit 139 继续单独标记 BLOCKED。Milestone 1B 候选库为 `/data/InfiniTensor/phase1b-cache-vector3`。

## 常用命令

```bash
cd /data/InfiniTensor/InfiniLM
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/caching.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/attention.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/provider.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/vectorized.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/common/test_summarize.py
```

候选库位于持久目录 `/data/InfiniTensor/phase1b-cache-vector3`。构建时使用 `build/caching.py --base-prefix /data/InfiniTensor/phase02-fixed-sm89`，避免旧 build object 混入；示例：

```bash
INFINI_ROOT=/data/InfiniTensor/phase1b-cache-vector3 \
LD_LIBRARY_PATH=/data/InfiniTensor/phase1b-cache-vector3/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/caching.py \
--threads 32 --grouped-heads 1 --vector --dtype fp16 --output test/bench/decode_ops/runs/m1b_vector_fp16_32
```

候选构建脚本在 `build/caching.py`，不把编译产物写入仓库。完整 A/B、Graph、sanitizer 命令见 `reports/phase0_2_commands.md`；Milestone 1A 结果见 `reports/milestone1A_report.md`；本轮补测与修复见 `reports/milestone1A_1_report.md`；Milestone 1B 见 `reports/milestone1B_report.md`；Milestone 1C grouped CTA 见 `reports/milestone1C_report.md`。

## 迁移映射

| 历史入口 | 新入口 |
|---|---|
| `phase0/correctness_support.py` | `decode_ops/common/correctness_support.py`（旧文件为兼容 wrapper） |
| `phase0/gqa_regression.py` | `decode_ops/correctness/attention.py` |
| `phase0/provider_replay.py` | `decode_ops/correctness/provider.py` |
| `phase0/split_control_probe.py` | `decode_ops/correctness/caching.py` |
| `phase0/caching_benchmark.py` | `decode_ops/benchmarks/caching.py` |
| `phase0/build_caching_isolated.py` | `decode_ops/build/caching.py` |
| `phase0/phase0_2A_report.md` | `decode_ops/reports/phase0_2A_report.md` |

历史 `phase0/runs/` 与 `phase0/phase0*_runs/` 原位保留，新的 JSON/log 仅放在 `decode_ops/runs/` 并由局部忽略规则精确放行。

Milestone 1C E/F 显式候选：`--threads 64 --grouped-heads 2 --vector`（E），`--threads 128 --grouped-heads 4 --vector`（F）。不设置这些环境开关时生产默认路径保持不变。

Milestone 2A attention 原型与复现命令见 `reports/milestone2A_report.md`；GQA-aware 路径通过 `INFINIOP_FLASH_GQA_SPLITKV=1` 显式启用，默认 dispatch 不变。

Milestone 2B shared-KV GQA Split-KV kernel、正确性矩阵和完整 attention 消融见 `reports/milestone2B_report.md`。实现通过 `INFINIOP_FLASH_GQA_SHARED_SPLITKV=1` 显式启用；复现脚本为 `correctness/gqa_shared.py` 与 `benchmarks/attention.py`，证据位于 `runs/m2b_*_20260906/`。当前 kernel correctness PASS、性能结论 NO_GAIN；shared-path memcheck/racecheck/synccheck PASS，两算子组合库集成已修复并 PASS（此前错误源于 host link 缺少当前 cache object），整模型 E2E 为 NOT_RUN。

Milestone 2C tile-level online softmax 优化见 `reports/milestone2C_report.md`；结果为 correctness/Graph/sanitizer/两算子集成 PASS，性能相对 2B 与普通 CTA 均 NO_GAIN，按停止条件不再扩展候选。

Milestone 3A CTA Split-KV 容量策略与 Provider 集成见 `reports/milestone3A_report.md`；shared-KV 分支已关闭。

Milestone 3B 见 `reports/milestone3B_report.md`；候选库为 `/data/InfiniTensor/phase3b-strategy`。`capacity_v1` 仅在显式 auto、目标 GQA/CTA/FP16-BF16 范围内启用，`num_splits=1` 明确走普通 `cta_nosplit`；provider 长序列、三轮 FP16/BF16 基准及 strategy memcheck 证据位于 `runs/m3b_*_20260906/`。生产默认环境变量为空时保持原 dispatch。
