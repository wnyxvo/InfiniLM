# decode_ops 实验入口

这里维护当前 decode correctness、paged caching 和 CUDA-event 实验。`test/bench/phase0/` 保留为历史入口与原始归档；新实验统一写入 `test/bench/decode_ops/runs/`。共享 native 绑定、独立 FP32 reference、判定和 provenance 位于 `common/`。

当前状态：Milestone 1A.1 补测已完成。驱动/NVML 已恢复；128/256/1024 的 FP16/BF16 correctness、InfiniCore Graph、负对照和两轮 CUDA-event 测量均通过。standalone caching 对 128/256 有稳定收益，但 append→attention 没有稳定端到端收益，默认 caching 仍为 1024。Phase 0.2 attention memcheck/racecheck 的历史 exit 139 继续单独标记 BLOCKED。修复后的候选库为 `/data/InfiniTensor/phase1a-cache-128-fixed`，由已验证 phase02 attention objects 作为 base。

## 常用命令

```bash
cd /data/InfiniTensor/InfiniLM
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/caching.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/attention.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/correctness/provider.py --help
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/common/test_summarize.py
```

候选库位于持久目录 `/data/InfiniTensor/phase1a-cache-128-fixed`。构建时使用 `build/caching.py --base-prefix /data/InfiniTensor/phase02-fixed-sm89`，避免旧 build object 混入；示例：

```bash
INFINI_ROOT=/data/InfiniTensor/phase1a-cache-128-fixed \
LD_LIBRARY_PATH=/data/InfiniTensor/phase1a-cache-128-fixed/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/decode_ops/benchmarks/caching.py \
--threads 128 --dtype fp16 --output test/bench/decode_ops/runs/fp16_128
```

候选构建脚本在 `build/caching.py`，不把编译产物写入仓库。完整 A/B、Graph、sanitizer 命令见 `reports/phase0_2_commands.md`；Milestone 1A 结果见 `reports/milestone1A_report.md`；本轮补测与修复见 `reports/milestone1A_1_report.md`。

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
