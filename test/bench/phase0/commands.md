# 复现命令

工作目录 `/data/InfiniTensor/InfiniLM`。先 `nvidia-smi` 确認GPU0空闲。使用已有venv，不安装或更新依赖。复跑请改output为新run_id，避免覆盖本次结果。

## 本次基线

```bash
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/native_baseline.py --quick --output test/bench/phase0/runs/smoke_20260905
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/native_baseline.py --quick --variant gqa --output test/bench/phase0/runs/gqa_repro_20260905
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/native_baseline.py --output test/bench/phase0/runs/native_20260905
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/native_baseline.py --quick --fp16 --variant gqa --output test/bench/phase0/runs/gqa_fp16_20260905
/data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/native_baseline.py --caching-only --aligned --output test/bench/phase0/runs/caching_aligned_20260905
/data/InfiniTensor/.venv-infini/bin/python /data/InfiniTensor/InfiniCore/test/infiniop/paged_attention.py --nvidia
/data/InfiniTensor/.venv-infini/bin/python /data/InfiniTensor/InfiniCore/test/infiniop/paged_caching.py --nvidia
```

stdout/stderr归档为各run的dispatch.log以及native主run下upstream日志。最初smoke遇FAIL退出；之后脚本改为记录FAIL继续通过的候选。当前脚本新增fp16/caching-only/aligned参数，主run默认语义不变。只比较同主run内数据。

## Provider、Graph、profiler

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=/tmp/phase0-provider-native-v2 /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --output test/bench/phase0/runs/native_20260905/provider_native.json
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=/tmp/phase0-provider-flash /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --flash --output test/bench/phase0/runs/native_20260905/provider_flash.json
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=/tmp/phase0-provider-nodes /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --output /tmp/phase0-provider-nodes.json
nsys export --type sqlite --output /tmp/phase0-provider-nodes.sqlite /tmp/phase0-provider-nodes.nsys-rep
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv /tmp/phase0-provider-native-v2.nsys-rep
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv /tmp/phase0-provider-flash.nsys-rep
ncu --target-processes all --kernel-name 'regex:pagedCaching' --launch-count 1 --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --csv /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --output /tmp/phase0-ncu-check.json
compute-sanitizer --tool memcheck --error-exitcode 99 /data/InfiniTensor/.venv-infini/bin/python test/bench/phase0/provider_replay.py --output /tmp/phase0-sanitizer-check.json
```

Ncu实际被计数器权限拒绝；不修改系统权限。第一次provider脚本导入module误作callable，已修正，失败trace不作为证据。完整trace在/tmp，版本控制仅保留小型摘要。`summarize.py` 是本次固定日志归档助手，不是通用测试入口；生成summary.csv并将观测dispatch写回raw.csv。

## 后续 E2E（当前 BLOCKED）

先准备venv依赖和_infinilm扩展，确认`import torch; import infinilm`成功。本轮缺transformers/pybind11/LM扩展，不安装。下列先加载torch的命令已尝试，失败日志保留；不是性能数据：

```bash
PYTHONPATH=python /data/InfiniTensor/.venv-infini/bin/python -c 'import torch,runpy; runpy.run_path("examples/bench.py",run_name="__main__")' --device nvidia --model /root/huggingface/models/Qwen--Qwen3-4B/snapshots/master --tp 1 --enable-paged-attn --attn default --batch-size 1 --input-len 256 --output-len 32 --warmup
```

恢复环境后B={1,16}、L={256,2048}、输出32，每个点独立进程；Graph加`--enable-graph`，FA改`--attn flash-attn`，固定`--block-size 256`。较大点先检查报告显存预算与实测峰值。debug/profiler run与正式计时分开；现有bench计时边界仍需验证，不能拿一次输出作正式统计。

普通CTA必须显式设置`INFINIOP_FLASH_DECODE_KERNEL=cta INFINIOP_FLASH_DECODE_SPLITKV=0 INFINIOP_FLASH_GQA_FUSED=0`。auto为`INFINIOP_FLASH_DECODE_SPLITKV=auto`，当前会触发错误GQA路径，不建议生产使用。改knobs后重新capture/独立进程；勿设置INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS并声称真实device graph。
