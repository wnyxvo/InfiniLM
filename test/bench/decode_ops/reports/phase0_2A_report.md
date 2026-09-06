# Phase 0.2A：同步修复证据补齐与验收收尾

日期：2026-09-05。当前仓库：InfiniLM `79acc1967fc717c2174db5e637f737c5662681a5`，InfiniCore `bd17d120b97de7bdf88a7b79a102e7088f656856`。同步修复前基线为 InfiniCore `c033221b6e319d5de1a1a8560ff2040588f3a9dd`。

## 总体结论

**PARTIAL**。Phase 0.2 的同步根因、最小 patch 和固定输入 A/B 已得到可追溯证据；判定逻辑已修正；目标 GQA、append 和 Graph 功能回归通过。memcheck/racecheck 在本机仍由 compute-sanitizer 自身 SIGSEGV（exit 139），没有有效工具诊断，因此 Sanitizer coverage 和总体 acceptance 不能标 PASS。按本轮边界不运行整模型 E2E、NCU 或性能实验。

## 分项判定

| 项目 | 判定 | 依据 |
|---|---|---|
| Archive completeness | **PARTIAL** | Phase 0.2 原始 run 文件存在并保留；清单 `phase02_runs/archive_0p2a/archive_inventory.json`；mem/race 的工具输出只有 sanitizer header，shell 日志在 `/tmp/phase02-memcheck.log`、`/tmp/phase02-race.log` |
| Verdict consistency | **PASS** | 汇总器独立推导 target/control/cache/overall，矛盾显式列出；3 个状态单测通过；Phase 0.1 派生汇总发现 3 条历史 graph/result 不一致，未修改原件 |
| A/B provenance | **PASS** | c033 与 bd17 独立 worktree；相同 CUDA recipe、sm_89、flags、链接输入；A/B 库独立路径和 build manifest；input/reference hash 相同 |
| Fixed-input correctness | **PASS** | A 74/100（26 次 Split4 失败），B 100/100；第二组独立进程 50/50；输入/reference 每次不变 |
| Graph/append correctness | **PASS（范围内）** | GQA forced/auto 216/216；provider 12/12 正向，4/4 负对照检出；Graph 单独记录 |
| Sanitizer coverage | **BLOCKED** | synccheck exit 0/0 errors；memcheck、racecheck exit 139，工具自身崩溃，无 kernel 诊断 |
| Overall acceptance | **PARTIAL** | 关键功能和 A/B 已完成，但 sanitizer 关键证据缺失 |

## 原证据和归因勘误

Phase 0.2 原始 `results.json`、`manifest.json`、`build.json` 和普通日志均仍在本地；此前因根目录 `*.json`、`*.log` 被忽略，已增加精确 `phase02_runs` 白名单。历史文件没有覆盖，当前汇总写入独立派生目录。旧 Phase 0.1 唯一综合失败行的 target attention/eager 均 PASS，失败来自 Split4 对照；修正后的汇总器不再把“目标 PASS、对照 FAIL”统计成正向综合 PASS。

## 严格 A/B 与固定输入

A：`/data/InfiniTensor/phase02-A-sm89/lib/libinfiniop.so`，由 c033 worktree 构建，SHA256 `013ca29f600c0142de668e0febbe6cd3181a2ea613aaf4852c94730572a7572a`。  
B：`/data/InfiniTensor/phase02-fixed-sm89/lib/libinfiniop.so`，由 bd17 worktree 构建，SHA256 `c4295408576a183747f8f1060a195143726866a0399ecef1e5e82e97ee71a4a6`。

A/B 固定 FP16 输入为 B=16、Hq=32、Hkv=8、D=128、page=256、lengths=`[1,7,8,9,255,256,257,2049] * 2`。两次独立进程的 input hash 均为 `950e08adc73f6a52d4f35d08e281cf69151ce611115fce9fbce712903c0cbcb9`，reference hash 均为 `07213cb04829fe47c3c104d8beeaa8d6ed37b1a7b16c827086a2f5a69dcdd894`。每一行记录 output hash、failed batch/head/dim、actual/expected/误差阈值，并验证 inputs/reference unchanged。

第二组 lengths=`[23,24,25,511,512,513,1023,1024] * 2` 在独立进程中 B 50/50 PASS。完整结果和派生 verdict 在 `phase02_runs/A_c033_*`、`B_bd17_*`、`fixed_split4_second_50_*`。

## 判定模型修复

`summarize.py` 现在输出 `target_correctness`、`control_correctness`、`cache_correctness`、`test_kind`、`negative_control_detected` 和派生 `overall_result`。正向测试只有 target、cache、必需 control 全 PASS 且 mode 已运行才可综合 PASS；目标 PASS + 对照 FAIL、目标 FAIL + 对照 PASS 均综合 FAIL；未运行综合 INCOMPLETE；负对照检出只计入 negative-control detection。已记录 result 与派生结果不一致时写入 `inconsistencies`，不修改 raw。

## 覆盖和 sanitizer

GQA bounded forced/auto 的 FP16/BF16 eager 与 torch CUDA Graph 为 216/216 PASS；普通 CTA、Split2、Split4 均包含在矩阵。独立 native/InfiniCore provider Graph 为 12/12 正向 PASS，skip-append 和 corrupt-write 4/4 检出。合法 metadata 更新和实际 dispatch 均记录。

synccheck 使用目标 `flashAttentionDecodeHd128CtaGqa4`、`--launch-count 4`，工具 exit 0 且 `ERROR SUMMARY: 0 errors`，应用结果单独通过。memcheck/racecheck 同条件下 compute-sanitizer 进程 SIGSEGV（exit 139），只输出 `========= COMPUTE-SANITIZER`，没有可用错误报告；本轮将它们标为 BLOCKED，没有复用 Phase 0.1 的 sanitizer 通过结果。

## 变更和限制

InfiniCore 仅保留三个 CTA kernel 的消费者完成 barrier；未改变默认 dispatch、tile、split、线程映射、cache layout、epsilon、精度或公开 API。InfiniLM 变更为 probe hash/provenance、汇总器、归档白名单、命令和报告。受影响的 hd64/128/192/256/576 及 MLA hd576 编译单元均纳入隔离构建。

仍未完成：稳定工具版本下的 memcheck/racecheck、空 split 的更广覆盖、完整模型 E2E、NCU 和性能计时。当前仅判断已具备进入 Paged KV Update CTA Right-sizing 的 correctness 前置条件；本轮不开展该阶段。

建议 commit message：

- InfiniCore：`fix(paged-attention): synchronize consumers before cp.async stage reuse`
- InfiniLM：`test(phase0): make phase02 evidence and verdicts reproducible`


## 持久化产物说明

云 GPU 的 `/tmp` 为临时目录。本次隔离库已复制到 `/data/InfiniTensor/phase01-fixed-sm89`、`/data/InfiniTensor/phase02-A-sm89` 和 `/data/InfiniTensor/phase02-fixed-sm89`；历史 manifest 保留原始 `/tmp` 运行路径和 hash，重启后应使用这些持久路径。
