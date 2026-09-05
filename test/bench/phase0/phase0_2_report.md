# Phase 0.2：失败归因与最小同步修复

日期：2026-09-05。基线为 InfiniLM `210a20876615ddad287be01e757f36815e842be1`、InfiniCore `c033221b6e319d5de1a1a8560ff2040588f3a9dd`；保留 Phase 0.1 的 GQA owner-thread normalization 修复。

## 结论

**PASS（本轮目标）**：Split4 失败根因已由固定输入复现、shared-memory 访问关系和单变量 A/B 支持；最小同步修复后同配置 100/100、第二组 page-boundary 输入 50/50，GQA bounded forced/auto 为 216/216。Provider 的独立 append/reference 与 Graph 正向、负对照也通过。

这不表示数学上证明不存在竞争；它表示在规定输入和重复次数下，修复消除了已观察到的失败。`compute-sanitizer` 的 memcheck/racecheck 本轮自身 SIGSEGV（exit 139）而无诊断，不能算通过；synccheck exit 0、报告 0 errors。整模型 E2E 和 NCU 仍按 Phase 0.2 说明不运行。

## Phase 0.1 归因更正

`post_fix_20260905T1237` 的 216 行中，目标 `attention_correctness` 全部 PASS，唯一综合失败是 FP16/mixed16/auto/eager 行的 `crosscheck_split4=FAIL`；该行 `eager_correctness=PASS`，不是 GQA 或 auto 输出失败。旧/修复库 Split4 比例（20/30、19/30）不能单独推断补丁恶化问题。

Provider 统计应分开：Phase 0.1 native 为 30 eager + 30 Graph 正向，20 个负对照；Flash 为 6 eager + 6 Graph 正向，4 个负对照。Phase 0.2 新 run 使用 6+6 正向和 4 个负对照。

## 固定复现

输入为 FP16、B=16、Hq=32、Hkv=8、D=128、page size=256，固定 seed、q/K/V、页表、lengths=`[1,7,8,9,255,256,257,2049] * 2`，每次调用前输出填充 NaN，记录 actual dispatch、输入/reference 结构和运行库 hash。

- 旧库 `/root/.infini/lib/libinfiniop.so`：100 次中 69 PASS、31 FAIL；失败为 `splitkv_cta`，非有限值为 0，但少数 head 的误差约 0.02。
- 修复库 `/data/InfiniTensor/phase02-fixed-sm89/lib/libinfiniop.so`：同一输入 100/100 PASS。
- 第二组 lengths=`[23,24,25,511,512,513,1023,1024] * 2`，独立进程 50/50 PASS。

结果、manifest、dispatch 和脚本快照在 `test/bench/phase0/phase02_runs/`。复现命令见 [`phase0_2_commands.md`](phase0_2_commands.md)。

## 根因与 patch

三个 NVIDIA CTA kernel（普通 CTA、Split-KV CTA、GQA CTA）都采用 `STAGES=3` 的 shared `sh_k/sh_v` 环形 buffer。当前 tile 的所有线程先读取 `sh_v[buf]` 累加；原代码随后直接由各线程向同一个 `buf` 发起 `cp.async`，写入 `tile_idx + STAGES`。原有 `cp.async.wait_group` 只保证异步 copy 完成后可读，不能保证此前所有消费者已经结束读取，因此缺少“消费者完成读取 → 生产者覆盖 buffer”的顺序保证。

在每个 prefetch 点前加入与既有线程参与关系一致的 barrier：`NUM_WARPS==1` 使用 `__syncwarp()`，否则使用 `__syncthreads()`；之后保留原有 commit、wait-group 和 drain。没有改变 dispatch、split、tile、CTA 映射、cache layout、epsilon 或精度。头文件影响的全部 NVIDIA paged-attention 编译单元（hd64/128/192/256/576、MLA hd576）均重新编译；隔离构建 manifest 在 `build_sm89_20260905T1410`，新库 hash 为 `c4295408576a183747f8f1060a195143726866a0399ecef1e5e82e97ee71a4a6`。

## 回归结果

| 路径/检查 | 结果 |
|---|---|
| GQA forced/auto，FP16/BF16，eager + torch CUDA Graph | 216/216 PASS（eager 54、Graph 162） |
| Split4 原失败点 | 100/100 PASS |
| 第二组 page boundary | 50/50 PASS |
| 普通 CTA、Split2 对照 | 包含于 GQA bounded matrix，均 PASS |
| Native/InfiniCore provider Graph | 12/12 正向 PASS |
| skip-append / corrupt-write | 4/4 被独立 validator 检出，记录为负对照 PASS |
| Phase 0.1 memcheck/racecheck | 0 errors / 0 hazards；本轮同工具启动 SIGSEGV，不能复用为本轮通过 |
| 本轮 synccheck | 工具 exit 0、0 errors；应用正向结果单独记录并通过 |

Append reference 在调用前从原 cache 克隆，按 slot/page table 更新 expected K/V，再用独立 FP32 reference 比较；整 cache 以 bitwise 方式检查 untouched 区域。Graph 使用同一 Graph 的合法 metadata 更新，不以 eager 结果推断 Graph。

## 限制与阶段判断

- 本轮 memcheck/racecheck 是 compute-sanitizer 进程自身 SIGSEGV（exit 139），没有 sanitizer 错误报告；需要后续在稳定工具版本/更小 kernel 选择下复核。
- 尚未运行完整 InfiniLM 模型 E2E；Phase 0.2 明确不要求它。NCU 仍受 `ERR_NVGPUCTRPERM` 阻塞。
- 空 split、全部 attention 变体和真实多层模型仍需后续覆盖。

当前具备进入 **Paged KV Update CTA Right-sizing** 的 correctness 前置条件，但本轮在此停止；不得把同步修复当作性能优化，也不进入 GQA-aware Split-KV 原型。

建议提交信息：

- InfiniCore：`fix(paged-attention): synchronize consumers before cp.async stage reuse`
- InfiniLM：`test(phase0): correct split-kv attribution and add phase02 repro`


## 持久化产物说明

云 GPU 的 `/tmp` 为临时目录。本次隔离库已复制到 `/data/InfiniTensor/phase01-fixed-sm89`、`/data/InfiniTensor/phase02-A-sm89` 和 `/data/InfiniTensor/phase02-fixed-sm89`；历史 manifest 保留原始 `/tmp` 运行路径和 hash，重启后应使用这些持久路径。
