# Phase 0.1：GQA correctness 与 append 可靠性报告

日期：2026-09-05  
仓库：`InfiniLM` HEAD `f1498ccd13d42f20c4ab5fe9e97e5bf70761fcdc`；`InfiniCore` HEAD `46ca684929aaa2ce69fb1b288787e8a859e26c93`。GPU 为 RTX 4090（CC 8.9，CUDA 12.8，torch 2.9.1+cu128）。生产源码只改动 `InfiniCore/src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh`；LM 侧改动均为测试、归档和报告工具。

## 结论

修复前的 GQA 错误已独立复现。修复后，独立 append reference、native/Flash provider、InfiniCore Graph、memcheck、racecheck 和 30 次稳定性回归通过；完整 bounded matrix 为 215/216 PASS。仍有一个完整回归中观察到的 FP16 `auto/eager` 失败（同一 mixed16 诊断复跑通过），`synccheck` 运行在 instrumentation 下出现数值失败，且既有 Split-KV（Split4）在旧库和修复库都存在残余失败。因此本阶段总体为 **PARTIAL**，不进入下一阶段优化；需先收敛剩余数值问题。

## 修复

GQA fused kernel 中，线程 0（每个 CTA 的 owner）更新了所有分组的私有 `m[g]/l[g]`，但原代码让 `tid < NGROUPS` 的多个线程直接读取各自未更新的 `l[g]` 来发布 normalization。修复将 `inv_l_shared[g]` 的发布统一放到 owner（`tid == 0`）并在原有 `__syncwarp/__syncthreads` 后使用，保持 epsilon、精度、dispatch、CTA 映射、tile、cache layout、API 和 split count 不变。

修复库通过隔离 sm_89 构建生成于 `/tmp/phase01-fixed-sm89/lib/libinfiniop.so`，SHA256 `004e96510800cdebdd489f1eed641a45f26593a5084befc3af6b17afbe98ed2f`；旧 `/root/.infini/lib/libinfiniop.so` 未覆盖。复现实验环境使用：

```text
INFINI_ROOT=/tmp/phase01-fixed-sm89
LD_LIBRARY_PATH=/tmp/phase01-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64
```

## 可靠性与独立性

`provider_replay.py` 在调用前克隆 cache，按 slot/page table 独立更新 expected K/V，并用 FP32 attention reference 比较输出；cache 用 int16 bit view 检查整块（包括 untouched region）。每个 variant 分开记录 eager 与 InfiniCore framework Graph，记录实际 dispatch 和哨兵输出。`skip_append` 与 `corrupt_written` 两类负对照均被 validator 检出。`summarize.py` 拒绝缺失输入、拒绝覆盖已有汇总，并将历史复用 eager 结果的 graph 行标为 `UNVERIFIED_LEGACY_GRAPH`。

Phase 0 原始 JSON/log 未改写；归档清单和 SHA256 在 `phase01_runs/archive_20260905T1238/archive_inventory.json`。运行时脚本 hash 对旧实验未知，未用当前脚本补造旧 manifest。

## 结果矩阵

| 检查 | 结果 | 证据 |
|---|---:|---|
| 修复前 GQA | **REPRODUCED**（manifest exit 1） | `pre_fix_20260905T1218` |
| 修复后 bounded GQA | **PARTIAL**，215/216 PASS；另有 `gqa_stability` 30/30 PASS | `post_fix_20260905T1237`, `gqa_stability_20260905T1305`, `mixed16_diagnostic_20260905T1243` |
| 独立 append native | **PASS**，80/80（40 eager、40 InfiniCore Graph） | `provider_fixed_20260905T1240` |
| Flash provider | **PASS**，16/16（8 eager、8 Graph） | `provider_flash_20260905T1255` |
| 负对照 | **PASS**，skip-append 12/12、corrupt-write 12/12 被检出 | provider runs |
| CUDA Graph / Nsight | **PASS**，Graph replay 有 GQA kernel；8 graph launches | `nsys_gqa_20260905T1300/graph_nodes.csv` |
| Compute Sanitizer memcheck | **PASS**，0 errors | `memcheck_20260905T1246` |
| Compute Sanitizer racecheck | **PASS**，0 hazards | `racecheck_20260905T1252` |
| synccheck | **PARTIAL/FAIL**：无 hazard 报告，但 instrumentation 下出现 attention 数值失败 | `synccheck_20260905T1246` |
| 既有 InfiniCore paged tests | **PASS** | `upstream_fixed_20260905T1255` |
| Split-KV Split4 | **残余问题**：旧库 20/30，修复库 19/30；不是本 patch 引入或解决 | `split_old_probe_20260905T1258`, `split_fixed_probe_20260905T1258` |
| 整模型 E2E | **NOT_RUN** | startup `bench.py --help` 已 PASS；未运行模型生成闭环 |
| NCU 指标 | **BLOCKED** | `ERR_NVGPUCTRPERM`；未修改驱动或权限 |

完整 manifest、results、dispatch、Graph 和 sanitizer 输出均位于 `test/bench/phase0/phase01_runs/`；复现命令见 [`commands.md`](commands.md)。

## 阶段判定

Phase 0.1 的修复和独立验证可以作为后续工作的 correctness prerequisite，但总体 gate 仍是 **NEED_MORE_DATA / NO-GO for GQA-aware Split-KV prototype**。先解决 Split4 残余数值失败、确认 synccheck 下的行为并完成受控 InfiniLM E2E，再讨论 Paged KV Update CTA right-sizing 或其他性能实验。本轮没有实施性能优化，也没有跨 GPU/跨轮报告加速比。

建议提交信息：

- InfiniCore：`fix(paged-attention): publish GQA normalization factors from owner thread`
- InfiniLM：`test(phase0): harden GQA and append reference provenance`
