# Phase 0.2 / 0.2A 复现命令

工作目录：`/data/InfiniTensor/InfiniLM`。已有 venv、CUDA 12.8、sm_89；不安装依赖、不下载模型。A 是 InfiniCore `c033221b6e319d5de1a1a8560ff2040588f3a9dd`，B 是 `bd17d120b97de7bdf88a7b79a102e7088f656856`。A/B 已复制到持久路径 `/data/InfiniTensor/phase02-A-sm89/lib/libinfiniop.so` 和 `/data/InfiniTensor/phase02-fixed-sm89/lib/libinfiniop.so`；历史 manifest 仍保留当时实际使用的 `/tmp` 路径。重建时必须使用新的输出目录，避免覆盖已有产物。

```bash
cd /data/InfiniTensor/InfiniLM
git -C /data/InfiniTensor/InfiniCore worktree add --detach /tmp/phase02-core-A c033221b6e319d5de1a1a8560ff2040588f3a9dd
ln -s /data/InfiniTensor/InfiniCore/build /tmp/phase02-core-A/build
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/build_phase02_isolated.py \
  --core /tmp/phase02-core-A --prefix /data/InfiniTensor/phase02-A-sm89-rerun \
  --evidence /data/InfiniTensor/InfiniLM/test/bench/phase0/phase02_runs/build_A_c033_rerun
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/build_phase02_isolated.py \
  --core /data/InfiniTensor/InfiniCore --prefix /data/InfiniTensor/phase02-fixed-sm89-rerun \
  --evidence /data/InfiniTensor/InfiniLM/test/bench/phase0/phase02_runs/build_bd17_rerun
sha256sum /data/InfiniTensor/phase02-A-sm89/lib/libinfiniop.so /data/InfiniTensor/phase02-fixed-sm89/lib/libinfiniop.so
```

```bash
INFINI_ROOT=/data/InfiniTensor/phase02-A-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-A-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/split_control_probe.py \
--variant default --repeats 100 --output test/bench/phase0/phase02_runs/A_c033_split4_100_20260905T1630
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/split_control_probe.py \
--variant default --repeats 100 --output test/bench/phase0/phase02_runs/B_bd17_split4_100_20260905T1640
```

```bash
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/gqa_regression.py \
--output test/bench/phase0/phase02_runs/gqa_fixed_20260905T1440
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py \
--variant gqa --output test/bench/phase0/phase02_runs/provider_fixed_20260905T1450
```

```bash
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=flashAttentionDecodeHd128CtaGqa4 --launch-count 4 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --variant gqa \
--output test/bench/phase0/phase02_runs/memcheck_20260905T1500
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=flashAttentionDecodeHd128CtaGqa4 --launch-count 4 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --variant gqa \
--output test/bench/phase0/phase02_runs/racecheck_20260905T1510
INFINI_ROOT=/data/InfiniTensor/phase02-fixed-sm89 LD_LIBRARY_PATH=/data/InfiniTensor/phase02-fixed-sm89/lib:/root/.infini/lib:/usr/local/cuda/lib64 \
compute-sanitizer --tool synccheck --error-exitcode 99 --kernel-name kns=flashAttentionDecodeHd128CtaGqa4 --launch-count 4 \
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --variant gqa \
--output test/bench/phase0/phase02_runs/synccheck_20260905T1520
```

```bash
python test/bench/phase0/summarize.py --run-dir test/bench/phase0/phase02_runs/A_c033_split4_100_20260905T1630 --output test/bench/phase0/phase02_runs/A_c033_summary
python test/bench/phase0/summarize.py --run-dir test/bench/phase0/phase02_runs/B_bd17_split4_100_20260905T1640 --output test/bench/phase0/phase02_runs/B_bd17_summary
python test/bench/phase0/test_summarize.py
cat test/bench/phase0/phase02_runs/archive_0p2a/archive_inventory.json
```

memcheck/racecheck 在本机 compute-sanitizer 进程启动后 SIGSEGV（exit 139），synccheck exit 0 且报告 0 errors；工具状态与应用 correctness 分开判定。
