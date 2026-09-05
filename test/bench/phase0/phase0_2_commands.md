# Phase 0.2 复现命令

工作目录：`/data/InfiniTensor/InfiniLM`。使用已有 venv，不安装依赖、不下载模型。修复库由 `/tmp/phase02-fixed-sm89` 提供；旧库为 `/root/.infini/lib/libinfiniop.so`。两者均记录在对应 run 的 `manifest.json`。

```bash
# 旧库：固定 Phase 0.1 Split4 输入，100 次
INFINI_ROOT=/root/.infini LD_LIBRARY_PATH=/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/split_control_probe.py \
  --variant default --repeats 100 --output test/bench/phase0/phase02_runs/baseline_split4_100_20260905T1400

# 修复库：同输入 100 次；第二组 page-boundary 输入独立进程
INFINI_ROOT=/tmp/phase02-fixed-sm89 LD_LIBRARY_PATH=/tmp/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/split_control_probe.py \
  --variant default --repeats 100 --output test/bench/phase0/phase02_runs/fixed_split4_100_20260905T1420
INFINI_ROOT=/tmp/phase02-fixed-sm89 LD_LIBRARY_PATH=/tmp/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/split_control_probe.py \
  --variant default --repeats 50 --seed 20260906 \
  --lengths 23,24,25,511,512,513,1023,1024,23,24,25,511,512,513,1023,1024 \
  --output test/bench/phase0/phase02_runs/fixed_split4_second_50_20260905T1430

# GQA forced/auto，native eager + torch CUDA Graph
INFINI_ROOT=/tmp/phase02-fixed-sm89 LD_LIBRARY_PATH=/tmp/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/gqa_regression.py \
  --output test/bench/phase0/phase02_runs/gqa_fixed_20260905T1440

# 独立 append reference、InfiniCore Graph、负对照
INFINI_ROOT=/tmp/phase02-fixed-sm89 LD_LIBRARY_PATH=/tmp/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
  /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py \
  --variant gqa --output test/bench/phase0/phase02_runs/provider_fixed_20260905T1450

# sanitizer（工具退出码与应用结果分开记录）
compute-sanitizer --tool synccheck --error-exitcode 99 --kernel-name kns=flashAttentionDecodeHd128CtaGqa4 --launch-count 4 .../provider_replay.py --variant gqa --output phase02_runs/synccheck_20260905T1520
compute-sanitizer --tool memcheck  .../provider_replay.py --variant gqa --output phase02_runs/memcheck_20260905T1500
compute-sanitizer --tool racecheck .../provider_replay.py --variant gqa --output phase02_runs/racecheck_20260905T1510
```

`...` 代表同一 Python 绝对路径及同一 `INFINI_ROOT/LD_LIBRARY_PATH`。本次 memcheck/racecheck 在启动后由 compute-sanitizer 自身 SIGSEGV（exit 139），无 sanitizer 诊断；synccheck exit 0 且报告 0 errors。不要把工具崩溃当作 kernel 通过。
