# Phase 0 — CUDA Decode 审计与基线

日期：2026-09-05。范围：源码审计、正确性基线、有限实验和路线判断。未修改生产 kernel、dispatch、API、cache layout 或 scheduler；未安装依赖、更新子模块、commit/push。报告后停止。

最重要的结果是 **现有 hd128 GQA fused kernel 有可复现的归一化正确性错误**。BF16、FP16 独立进程复现；每组第 0 个 Q head 正常，第 1–3 个错误。现有 auto 在若干代表点选中此路径，不能作为正确的优化基线。优先修复并回归该问题，再讨论性能。KV update right-sizing 有明确源码依据，但尚无修改后的收益数据；GQA-aware Split-KV 当前 NO-GO。

证据标记：SOURCE_CONFIRMED 是当前源码；RUNTIME_CONFIRMED 是执行路径；MEASURED 是实际数据；HYPOTHESIS 是待验证解释；BLOCKED/NOT_RUN 不表示通过。下文源码路径以 `LM=/data/InfiniTensor/InfiniLM`、`Core=/data/InfiniTensor/InfiniCore` 为根，行号对应本次 checkout。

## 1. 仓库、环境与产物来源

| 项目 | 事实 |
|---|---|
| LM | `/data/InfiniTensor/InfiniLM`，`08ff3fe88fafce72f7fc485b87885d7135d6207b`，分支 feat |
| Core | `/data/InfiniTensor/InfiniCore`，`46ca684929aaa2ce69fb1b288787e8a859e26c93`，分支 feat |
| 初始状态 | 两仓库 `git status --short` 均为空；最终仅 LM 增加 `test/bench/phase0/` |
| 指令文件 | 检查两个仓库及父目录，未发现适用 AGENTS.md；读取双方 README、LM setup.py/xmake.lua 和 Core 构建配置 |
| GPU | GPU 0，RTX 4090，128 SM，CC 8.9，nvidia-smi 总显存 24564 MiB，Torch 可见 24330 MiB；初始 2 MiB、0% utilization、无计算进程 |
| Driver/toolkit | Driver 570.124.06；nvcc 12.8.61；g++ 13.3.0 |
| 系统 Python | `/usr/bin/python`，3.12.3；系统 torch 2.6.0a0+ecf3bae40a.nv25.01；没有 infinicore/infinilm 包 |
| 实验 Python | `/data/InfiniTensor/.venv-infini/bin/python`，torch **2.9.1+cu128**；所有正式数据来自此解释器，不与系统 torch 数据混合 |
| Core Python | editable 源码包 `/data/InfiniTensor/InfiniCore/python/infinicore` |
| 本地模型 | `/root/huggingface/models/Qwen--Qwen3-4B/snapshots/master/config.json`：32 Q heads、8 KV heads、D128、36 layers、BF16、max position 40960；safetensors 文件合计 8,044,982,000 bytes |
| profiler | nsys 2024.6.2；ncu 2025.1.0；Compute Sanitizer 2025.1.0 |

子模块完整版本见每个 run 的 manifest。主要版本：Core InfiniOps `47c1c4969c4312b4f15fc73928c72e4c262f79ce`、InfiniRT `b14539d0bdc3c4b79384d3c8805cbe915e40ad91`、InfiniCCL `c21a5b465dbdd94bdc246e32cf9c985f2ca01c9e`、FlashAttention `10846960ca0793b993446f6dbaf696479c127a9d`、CUTLASS `087c84df83d254b5fb295a7a408f1a1d554085cf`。mate 未初始化，未更新。LM json `5ed07097faa6c50199c4a3b66e5ed37d4fbfccc2`、spdlog `88a0e07ad5bb3e2651cd5613530b3f06a15fc400`。

Core `.xmake/linux/x86_64/xmake.conf`：release、sm_89、nv-gpu=true、cpu=true、aten=true、graph=true、flash-attn 指向本地 third_party，**infiniops=false**、ccl/cudnn=true、ninetoothed=false。这是构建配置文件事实，不是独立的二进制构建证明。`xmake --version` 被其 root 策略拒绝；未绕过该策略，版本 NOT_CONFIRMED。

RUNTIME_CONFIRMED：进程 maps 显示 `/root/.infini/lib/libinfiniop.so`、`libinfinirt.so`、`libinfinicore_cpp_api.so`、`libinfiniccl.so`、`libflash-attn-nvidia.so`。加载 FlashAttention 库本身不表示调用了 FA kernel。native 路径由 debug 和 Nsight 进一步确认。

安装的 libinfiniop 与 `Core/build/linux/x86_64/release/libinfiniop.so` SHA256 一致：`c659030a2ea2b51c8486ae18635ffe8d8273b79d12c81c76b7b0bc577c35a58e`。证明运行的是本地已有产物的相同字节；**没有嵌入 SHA 或完整构建记录，不能严格证明产物由当前 HEAD 完整重建**。本轮没有重建。C++ provider 的路径验证限定为当前加载二进制的行为。

仅采集 CUDA_HOME、LD_LIBRARY_PATH、INFINI_ROOT、CUDA_VISIBLE_DEVICES、PYTHONPATH、VIRTUAL_ENV、相关 Graph/FLASH knobs；原始选择见 manifest，无凭据。实验开始这些 dispatch knobs 未设置；脚本每 variant 清除并显式设置自身 knobs。没有修改全局环境。

基础工具 sandbox 因 `bwrap: No permissions to create new namespace` 无法启动；工具请求获得许可后在沙箱外完成读取/测试/报告写入，没有绕过审批。报告放在允许写入的仓库目录。

## 2. Provider 调用链

SOURCE_CONFIRMED：LM `csrc/layers/attention/backends/attention_layer.cpp:14` 的构造器选择 static/paged/flash；`examples/bench.py:482` 仅在 `enable_paged_attn && attn_backend == default` 时转换为 paged-attn。因此 `--attn=default` 单独不能证明分页或 native。

`paged_attn.cpp:20` 的 `PagedAttentionImpl::forward` 先 `do_kv_cache_update`，decode 调 paged_attention_、prefill 调 paged_attention_prefill_；`flash_attn.cpp:30` 的 `FlashAttentionImpl::forward` 同样先更新 KV，decode 将 q view 成 `[B,1,Hq,D]` 后调 mha_kvcache；其 cache update 对 BSHD cache permute 为 BHSD。view/permute 本身是元数据转换，不能记作实际 memcpy。

| 分页配置 | KV update | Attention | 限制、fallback 与本次证据 |
|---|---|---|---|
| default/paged + InfiniOps OFF | native `infiniopPagedCaching` | native `infiniopPagedAttention` | C++ 两算子测试 RUNTIME_CONFIRMED；LM 完整调用链仅源码审计 |
| default/paged + InfiniOps ON | 外部 `ReshapeAndCacheFlash::Call`，该 wrapper 没有 native fallback | 条件成立时外部 `FlashAttnWithKvcache::Call`，否则 native plan/run fallback | 当前 OFF 产物不能切环境变量变成 ON；NOT_RUN，不重建 provider 矩阵 |
| flash-attn + 当前 OFF | native caching，cache 视图转换 | `mha_kvcache` → 内置 `flash::mha_fwd_kvcache` | C++ provider eager + device graph 均 PASS；不是外部 InfiniOps 调用 |

精确入口：Core `src/infinicore/ops/paged_attention/paged_attention_infiniop.cc:16/43/66`（plan/run/register）；`paged_attention_infiniops.cc:29` canUseFlashAttention、`:92` plan、`:127` run；`src/infinicore/ops/paged_caching/paged_caching_infiniop.cc:15/34/55`；`paged_caching_infiniops.cc:19/33/38`（plan/run/ReshapeAndCacheFlash）。`Core/xmake.lua:745/748` 分别控制 ENABLE_INFINIOPS_API 和 LINKED_FLASH_ATTN_WITH_KVCACHE；不能以 InfiniOps ON 推定 FA 已链接。

外部 paged attention 的 guard 仍要求 page size 是 **256 的倍数**、q/out 3D、cache 4D 且 K/V 同 shape、FP16/BF16、D≤256 且 D%8=0、head 数整除、尾维 stride=1、block table/cache lens 为连续 I32，并且链接宏存在。`:106` 在 plan 固定 use_flash_attention，不是每步自动 fallback。ON caching 的支持范围由外部实现决定，wrapper 无上述 native fallback 保障。

`Core/src/infinicore/ops/mha_kvcache/mha_kvcache_flashattn.cc:36` 的 canUseInfiniOps 也核验 256 倍数；`:201` 后内置 FA 路径对 K/V contiguous，非连续输出需 copy back；`:239` 调 mha_fwd_kvcache，未启用 FA 时 `:273` 抛错。engine flash 的 BSHD dense cache 可免除 K/V contiguous 实际拷贝；不能推广到任意 view。

本次 C++ provider 测试 shape：q `[2,32,128]`、新 k/v `[2,8,128]`，cache native `[8,8,256,128]`，stride `[262144,32768,128,1]`；flash `[8,256,8,128]`，stride `[262144,1024,128,1]`；q stride `[4096,128,1]`。block table `[2,4]` I32、lens `[2]` I32、slots `[2]` I64。初始长度 255/257，三步更新至 256/258、257/259、258/260（**包含新 token**），随机新 K/V、一次重排合法物理页。TP=1、BF16、page=256。

Nsight native 实际名称：`pagedCaching<__nv_bfloat16,1024>`、`flashAttentionDecodeHd128SplitKvCta<int,__nv_bfloat16>`、`flashAttentionDecodeHd128SplitKvCombine<__nv_bfloat16>`；flash 实际名称：同一 native caching，加 `flash::flash_fwd_splitkv_kernel` 和 `flash_fwd_splitkv_combine_kernel`。详见 nsys CSV，不能把一个 operator 写成一个 kernel。

## 3. hd128 inventory、dispatch、容量

SOURCE_CONFIRMED：Core `src/infiniop/ops/paged_attention/nvidia/paged_attention_hd128.cu` 的 `launch_decode_hd128_impl:423`（参数与决策集中在 :451 后）；底层逻辑 `cuda/kernel_v2.cuh`。

| variant | grid / block | kernel_v2.cuh 函数起点 | 约束 |
|---|---|---|---|
| Warp no-split | `(Hq,B,1)` / 32 | flashAttentionDecodeWarpKernel:135 | hd128 FP16/BF16；支持 GQA mapping、ALiBi |
| CTA no-split | `(Hq,B,1)` / 32或64 | flashAttentionDecodeCtaKernel:1245 | tile 8/16；32线程每线程4维、64线程每线程2维；多阶段 shared/cp.async 已有 |
| GQA fused no-split | `(Hkv,B,1)` / **固定64** | flashAttentionDecodeCtaGqaKernel:1737 | GQA=4、无ALiBi、CTA、!split；固定 tile8，覆盖请求的普通 CTA tile/thread 选择；本次正确性 FAIL |
| Split warp | `(Hq,B,S)` / 32 | flashAttentionDecodeSplitKvWarpKernel:341 | S≤8；每 query head 独立 partial |
| Split CTA | `(Hq,B,S)` / 32或64 | flashAttentionDecodeSplitKvCtaKernel:621 | tile8/16，已有流水线 |
| Combine | `(Hq,B,1)` /32 | flashAttentionDecodeSplitKvCombineWarpKernel:557 | 每 split partial m/l/acc 合并，必须计入 latency |

GQA 多 Q 复用、Online Softmax、cp.async、多阶段预取、Split-KV 和 combine **全部是上游已有能力**，本轮个人新增只有测试/benchmark/分析。

descriptor：`nvidia/paged_attention_nvidia.cu:182` create，`:196` 后保留 8 splits workspace，即 `8×B×Hq×(Dv+2)×4` bytes；hd128 Hq32 时 B1=133120、B16=2129920、B32=4259840 bytes，即使 no-split 仍预留。partial acc `[S,B,Hq,128]`、m/l `[S,B,Hq]`；offset 按最大8 splits 预分配，combine 实际循环 S。空 split 写 m=-inf/l=0/acc=0（kernel_v2 :379、:672 附近）；本轮长度1的强制空 split 特例未额外跑，不能由源码宣称已通过。

`info.h:52/67/95/156` 检查 dtype、连续尾维、索引 stride 并存储 q/cache/out stride；普通 hd128 launch 没有独立传入 q head stride 或 block-table row stride，不能把 descriptor 可表达的所有 stride 当作已验证支持。测试只用连续 head、连续行页表；cache 外维 stride 与非对齐 KV update 在单独测试覆盖。

开关均在 **host calculate/launch 调用时 getenv**，不是每个 graph replay 重新读取：

| 环境变量后缀（INFINIOP_FLASH_） | 支持值 / 默认 / 实际含义 |
|---|---|
| DECODE_KERNEL | cta 默认；任何非 cta 字符串都选 warp，并非 auto kernel selector |
| GQA_FUSED | 默认 true；1/true 开，其余显式值关 |
| CTA_TILE | 8默认、16；其他保留8 |
| CTA_THREADS | 64默认、32；其他保留64 |
| DECODE_SPLITKV | 默认 true；0/false 关，1/true 开，auto 运行 heuristic，其他关 |
| NUM_SPLITS | 默认4；数值最终夹到1..8；auto 或非正整数使 fixed=false；与 SPLITKV=auto 同时使用时有后续再计算逻辑，不建议混合解释 |
| DEBUG_DISPATCH | 1/true 开；按签名变化打印，有抑制重复行为；split 数动态计算某些分支发生在打印后，不能只读一行解释所有组合 |
| DEBUG_SPLITS | 1/true 开；按容量估算变化打印，可能不打印 batch-only 变化 |

分支：读取 knobs → SPLITKV auto 决定 split → GQA fused guard → Split partial+combine → 普通 CTA/warp。**默认 use_split=true、num_splits=4，默认 GQA flag=true 不代表运行 GQA**。本次普通 CTA 明确设置 GQA_FUSED=0。

`:40` chooseNumSplitsHeuristic 用 waves×每块 token 工作量并估计 combine 一波；`:575` 输入 `max_num_blocks_per_seq × page_block_size`，不是每序列有效 cache_len，也不看 batch 长度分布。长度≤256 选1；B1/H32/SM128、估计8192 选4；B16/H32 等大量 CTA 时可选1。auto 中选1走 no-split，因此可能触发坏的 GQA。

实验证据：同有效 L256，页表宽度1时 auto→GQA FAIL；宽度32时 auto→Split4 PASS。相同宽度32，L17/L256/L8192 均 auto→Split4；混合 `[17,255,256,2049]`、B4、宽32，auto→GQA FAIL。各 case 的 cache 值并非跨 case 相同，因此**容量对照仅用于 dispatch 路径结论，不将 case 间 latency 差异完全归因容量**。同 case 各 variants 输入完全相同。

host 可信元数据已经存在：LM `PagedCompiler::get_compiled:233` 要求 CPU I32 total_sequence_lengths，`:247` bind_host_int_array；无需每层每步 D2H。将长度统计未来传给 native plan/dispatch 的 API/录图策略成本尚未验证；本轮未改 API。

## 4. GQA 正确性缺陷

SOURCE_CONFIRMED：`kernel_v2.cuh:1994` 仅 warp0/lane0 更新所有 `l[g]`，`:1997` 写线程私有数组；`:2105` 却让 tid<4 的各线程读取自己的 `l[tid]` 写 inv_l_shared。线程1–3的 l 始终为0，除以1e-6，导致分组后3 heads 输出放大。正确性结果与这个错误一致；未做生产修复/修复后对照，不把推断后的补丁称为验证完成。

MEASURED：`gqa_repro_20260905/dispatch.log` 独立 GQA BF16 B1/L256，head0/4/8… max abs error约0.00046–0.00098；其他 heads 最大到11,141,120。`gqa_fp16_20260905` 同一点后3 heads出现 inf，正常 heads误差约0.00006–0.00012。主矩阵7个 GQA 点均FAIL；auto在 b1l256、b16l2048、b32l8192、mixed 共4点FAIL。这不是普通 BF16 容差问题。

最小后续修复范围：仅 kernel_v2.cuh 的归一化结果广播/写出线程归属，随后覆盖 BF16/FP16、GQA4、多页、短页、多步 append/Graph；**本轮未实施**。在修复并证明正确前，auto/GQA latency 不排名。

## 5. Graph 真实性

SOURCE_CONFIRMED：LM `PagedCompiler:28` 收集 B1..63，64..112每16，128..224每32，256..512每64，并补最大batch；不是 context bucket。`:59` compile 用长度全1、页表逻辑宽度 **num_blocks总池容量**，holder 是 num_blocks×max_batch。`:169` startGraphRecording 录框架算子；真正 device capture 在 Core `src/infinicore/graph/graph.cc:158` Graph::instantiate：先5次 warmup，`:194` BeginCapture，`:220` EndCapture，`:227` Instantiate。

`graph.cc:91` Segment::run 更新注册 hook 后 launch device graph；无 device graph 则 host op replay。`:169` 只要存在 INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS（即使值0）就绕过 capture；`:175` 按 capture-safe 分 segment，可含 host segment。GraphManager start/stop 位于 `graph.cc:258` 后；stop_recording 调 instantiate。native paged attention/caching 只注册 plan/run/cleanup，**没有 replay update hook**。

LM `get_compiled:182` 在 replay 前 copy input_ids、position_ids、lengths、offsets、cu_seqlens、slots；`:213` 页表 active rows memset -1 后 copy runtime logical region；还有 runtime-state reset（Marlin 相关模型可能 memset，不应把 Qwen BF16 也武断算入）。只处理纯decode；batch未捕获、表宽超过捕获容量、Mamba metadata不匹配等回 eager。原生已捕获 kernel 的 variant/S 固定，长度张量内容可变，不会每次 replay 做 host auto dispatch。修改环境变量后必须重新capture；基线每variant新建图。

RUNTIME_CONFIRMED / MEASURED：provider_native.json 和 provider_flash.json 均 PASS。debug：2 operators / 1 segment / 0 host segments。Nsight native/flash 均看到 BeginCapture×1、Instantiate×1、GraphLaunch×3。额外 native node trace：**3次 replay 共9 kernel nodes**（caching、partial、combine 各3），图内 memcpy=0、memset=0，见 graph_evidence.json/graph_nodes.csv。图外确有测试输入更新及reference拷贝，不能当作 engine 开销。此结论限两个算子图，**不是整模型 PagedCompiler 已实测**。

native node 属性：caching grid(8,2,1)/block1024、32 registers/thread、static shared0；partial grid(32,2,4)/block64、68 registers/thread、static shared12388B；combine grid(32,2,1)/block32、38 registers/thread。来自 Nsight launch metadata，不是 occupancy 测量。

## 6. KV update mapping 与候选

Core `src/infiniop/ops/paged_caching/cuda/kernel.cuh:55` token=blockIdx.y、head=blockIdx.x；`nvidia/paged_caching_nvidia.cu:79` launchKernel 的 grid=(Hkv,num_tokens,1)，注释“1D one block per token”不准确。`:168` Descriptor::calculate 先按 maxThreadsPerBlock≥1024 选1024，之后512；后面的4096条件被前面覆盖。4090实际1024由Nsight确认。

Dk=Dv128时 tid0..127参与两段元素copy，其余896线程不搬K/V；128/1024=12.5%是参与拷贝线程比例，**不是 occupancy，也不表示8倍可实现加速**。相邻lane访问相邻D元素，所以标量源码访问已经合并；机器指令究竟16/32/128bit需 SASS/Ncu，NOT_CONFIRMED。源码逐元素赋值不等于确定的机器标量指令。

地址：源 `token×src_stride + head×src_head_stride + i`，目的 `physical_page×block_stride + head×head_stride + slot_offset×slot_stride + i`；物理页来自非负 slot/page_size。尾维隐含步长1，slot数组隐含连续；info.h没有为所有这些隐含条件做完整检查。未来fallback只能覆盖原API真正合法布局，不能声称支持任意stride。`info.h:44` FP16/BF16/FP32同dtype，slot I64；`:80` K cache D必须等于Dk、V cache D≥Dv，V padding保留；kernel :63 negative slot整CTA返回；不做越界slot验证，调用者需保证合法。

MEASURED：aligned KV BF16 D128 graph_batch（7样本×100，μs，中位数[min,max]）：T1 **1.505[1.495,2.130]**；T4 **1.495[1.494,1.935]**；T16 **1.638[1.628,2.088]**；T32 **2.458[2.458,2.888]**。T>1 的最后slot=-1，实际写入T-1个，避免与全有效decode混淆。另测尾维起始偏移1元素、head/token stride129、Dv96且V cache128、BF16/FP16/FP32，共24个组合，整个K/V cache bitwise与reference一致（含未写位置）；连续布局另24个组合也通过。微基线重复同一slot写入，是热cache，不是实际增长的多层访问。

HYPOTHESIS：先比较CTA right-sizing，再联合线程mapping做向量化，随后alignment-aware16B fast path与safe fallback。FP16 D128单K或V仅256B，16B/vector只需16线程覆盖；指令减少与有效并行度要共同衡量，不假设越宽越快。16B条件需实际基地址、storage offset、每头/每slot偏移、dtype、D、stride逐项验证。

未实现新kernel，因此相对倍数/绝对节省为 **NOT_MEASURED**。单层全decode和E2E占比 BLOCKED。只作理论预算：若每层一次同类KV写入，36×1.505≈54.2μs（T1）、36×2.458≈88.5μs（T32）是把本次热微基线线性外推的量级，并非实测engine上限。真实可省时间≤原组件时间；Amdahl全步上限1/(1-f)，f尚未知。

## 7. 有限 attention 性能结果

主 run_id **native_20260905**；原始 raw.csv（含每样本）、summary.csv、dispatch.log、manifest.json 均同目录。Hq32/Hkv8/D128/page256/BF16、scale=1/sqrt128、无ALiBi，随机物理页。reference独立FP32 QK、softmax、PV，GQA repeat-interleave映射；decode读取长度包含已追加token，无未来token。BF16 atol=.005/rtol=.05；FP16独立复现用.001/.01。

计时：10 warmup，7样本，每样本100调用。eager CUDA events位于显式提供的当前stream，但包含Python/ctypes enqueue空隙，是operator提交间隔，不能叫纯kernel。graph_batch捕获100 native calls，再用CUDA events测replay/100；包含Split partial+combine，排除分配、reference、capture和首次预热。**graph_batch不是InfiniLM的graph模式**。debug仅用于正确性单次dispatch，正式计时关闭；profiler runs不作为正式latency。

下表仅 **graph_batch μs**，默认列给min/max波动，其余完整波动见CSV：

| B / 有效长度 / 页表宽度 | 默认Split4 median[min,max] | auto | Warp | 普通CTA | Split2 | Split8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 /256 /1 | 8.724[8.714,9.728] | FAIL | 81.009 |23.060|13.998|6.513|
| 1 /8192 /32 |158.761[158.740,159.580]|158.740|2348.800|644.557|313.876|85.965|
|16 /2048 /8 |196.403[195.748,196.576]|FAIL|1004.319|217.784|218.173|184.503|
|32 /8192 /32 |1468.508[1434.133,1480.397]|FAIL|4090.378|2493.348|1662.607|1459.989|
|1 /256 /32 |8.131[8.120,9.001]|8.131|75.233|21.361|12.974|6.595|
|1 /17 /32 |3.850[3.850,4.659]|3.850|6.267|3.164|4.454|4.547|
|4 /混合 /32 |55.245[55.173,55.921]|FAIL|583.035|163.738|86.579|46.080|

GQA7点均FAIL，不展示其latency。所有展示时间的配置先通过reference。最佳固定配置只能称“本次已测候选”：Split8在7点中6点最快，L17普通CTA最快；没有定义业务权重，不选所谓全局最佳。代表点Split8对**同run同shape同dtype同graph_batch默认Split4**：B1L8192约1.847×，绝对72.80μs；B1L256约1.34×，绝对2.21μs；B32L8192仅约1.006×，差值落在波动范围，不宣称稳定收益。未扫描threads/tile及split warp组合，不称全variant oracle。

各variant按固定次序执行，无随机AB交错；没有锁频，热态/功耗漂移可能影响结果。原始样本保留，微秒点max明显受波动影响。本轮证据已足以暴露正确性优先级，因此停止细扫，不从这些数据拟合dispatch阈值。

## 8. 正确性、资源与 E2E

| 检查 | 结果/边界 |
|---|---|
| 现有 Core test/infiniop/paged_attention.py --nvidia | PASS，包含FP16/BF16、MHA及不同head dims；是底层native API，不是provider矩阵 |
| 现有 paged_caching.py --nvidia | PASS，包括FP32回归与Dk/Dv不同 |
| 新native attention七点 | 默认、warp、普通CTA、Split2/8 PASS；auto/GQA失败如上 |
| native CUDA Graph metadata | 每case重新捕获默认Split4，同一图更新合法长度/页表3次，PASS |
| C++ native append→decode Graph | PASS，3步，包括非整页和跨页，kernel/node证据齐全 |
| C++ flash append→decode Graph | PASS，3步，独立reference；没有与native作正式latency A/B |
| Compute Sanitizer memcheck | 短native provider 图0 errors；不是racecheck或全矩阵sanitizer |
| 额外覆盖未完成 | 强制空split长度1、ALiBi、新attention非对齐/非标准head stride、独立页表物理stride≠逻辑宽度、全模型Graph correctness；NOT_RUN |

资源公式：BF16 KV每token每层 `2(K,V)×8×128×2=4096B`，36层=147456B。page256每物理页全模型36MiB。模型磁盘权重约7.493GiB（估算常驻量，不代表实测峰值）；输出32 tokens，B1L256页向上取整至512，KV72MiB；B1L2048→2304，KV324MiB；B16L256→512，KV1.125GiB；B16L2048→2304，KV5.0625GiB。后者权重+KV约12.56GiB，再预留数GiB activations/workspace/allocator和多batch Graph保留。未测Graph峰值，因此只能作为待复测预算。

单层最大B32L8192 KV=1GiB，已运行；整模型该点即使不计输出/浪费也36GiB，**resource_excluded**。本轮没有触发OOM，也未下载模型。

E2E计划小/较大batch={1,16}，context={256,2048}，固定输出32，TP1/native eager/device graph，再视环境加入FA。**全部E2E数据 BLOCKED**。首先直接启动 examples/bench.py 在 import infinicore 遇 libtorch.so搜索问题；进程内先import torch排除此问题后，实际遇 `ModuleNotFoundError: transformers`。venv也没有pybind11，LM无_infinilm扩展，Xmake pybind11包缓存缺失；不能在“只用已有依赖、局部增量构建”范围直接完成官方构建。日志 e2e_attempt.log/e2e_preload_attempt.log 已保留。最小修复方案是另行准备匹配该venv的LM依赖和本地扩展构建；涉及依赖安装，按任务要求本轮报告阻塞而不安装。系统torch带pybind头文件不等同官方Xmake包依赖已就绪。

审计发现 examples/bench.py:137 的常规KV估算缺少K+V的×2，不能用它作OOM预算。`:390/404` time.time 包围generate，不含后续输出转numpy，且打印在范围内；warmup_steps=1后reset缓存，Graph首捕获是否被排除需实际核对。`python/infinilm/infer_engine.py:553/665` perf_counter每生成迭代，`:678/682` 用第一迭代作prefill TTFT、后N-1步平均作ITL；不是CUDA-event kernel计时，需确认下层同步边界。

后续口径：TPOT/ITL=decode段耗时/(Nout-1)，decode throughput=B×(Nout-1)/decode段时间；TTFT含prefill到首token，完整生成吞吐=B×Nout/完整generate时间，另报输入吞吐。不能把offline batch测量叫online serving。当前无实测TTFT/ITL/decode-step/E2E占比，不用最终文本一致替代数值校验。

## 9. Profiler 和限制

小型nsys CSV、graph_nodes.csv、graph_evidence.json已保留，完整trace在/tmp，未纳入版本控制。CPU perf_event不可用，因此nsys禁用CPU sampling；CUDA trace可用。Ncu实际返回 **ERR_NVGPUCTRPERM**，即使当前uid=root也没有目标GPU性能计数器权限；未修改驱动、capability或系统配置。load/store指令宽度、L2/DRAM吞吐、stall、achieved occupancy均 BLOCKED/NOT_MEASURED。不声称memory-bound或launch-bound。重复KV有缓存热度且4090 L2为72MiB；真实多层轮转需另测。

Profiler含reference、初始化、元数据更新；整体kernel百分比不是decode占比。native图内3节点/replay已单独分离，memcpy/memset节点为0；不能据此声称LM replay无copy（PagedCompiler源码明确有copy/memset）。FA节点级额外copy数量未做单独node trace归因，只有整体API/kernel证据，不把完整统计全算给provider。

## 10. 下一阶段判断与个人贡献边界

| 阶段 | 结论 | 后续范围与进入条件 |
|---|---|---|
| 正确性修复 | GO（建议，未实施） | kernel_v2.cuh GQA normalization ownership，独立FP32 reference和Graph回归后才恢复候选 |
| Milestone1 KV update | NEED_MORE_DATA；可进入限定实验 | paged_caching/nvidia launch right-sizing→cuda mapping/vectorization→alignment guard/fallback；记录绝对μs、倍数、适用范围与E2E占比，无收益保留负结果 |
| Milestone2 dispatch | NEED_MORE_DATA | 修复GQA后重测原默认、已有auto、最佳固定、每shape最佳有效variant；再用未调参长度/混合batch验证；区分eager与capture固定dispatch，不直接改默认 |
| Milestone3 GQA-aware Split-KV | **NO-GO，当前不进入原型** | 现有GQA基线错误、Ncu权限阻塞、E2E未完成，缺少“最佳已有GQA vs最佳Split”的公平证据 |

未来M3若重新评审：CTA按KV head/sequence/split映射，多个Q共享K/V只是候选；128thread/4warp并非既定设计。分析CTA数量减少、shared/register增加、同步、并行度、combine/workspace；不同实现允许各自最优S。现有L2可能已复用KV，global load减少不意味着DRAM降低4倍。复用combine需保持 `[S,B,Hq,D]` 排列和非归一化partial m/l/acc语义、空split。Graph context buckets、append fusion均延期；append集成还需核验RoPE、slots和长度定义，不能声称省去历史KV读取。

回答项目问题：

1. **最先改的是GQA fused归一化的线程归属正确性错误**，证据是BF16/FP16独立失败及源码一致的分组head模式。性能方向首选KV update CTA right-sizing做可证伪实验，尚无收益承诺。
2. right-sizing/向量化收益、memory/launch瓶颈、真实decode/E2E占比均只是HYPOTHESIS或未知；需新旧CUDA实现A/B、指令/计数器证据、合法布局回归、轮转KV和整模型验证。
3. **当前不值得直接进入GQA-aware Split-KV原型**；先修正确性并恢复公平基线。
4. 目前只有源码审计、缺陷复现、native基线与provider/Graph验证，**还不足以支撑独立CUDA优化简历项目**。需要本人实质CUDA修改、映射/访存解释、完整正确性与负结果、稳定绝对收益、profiler证据、未调参shape验证及InfiniLM E2E归因。上游能力、dispatch调参、框架接入、benchmark必须分开写。

本轮未增加生产trace，复用既有debug。全部GO仅为后续建议；到此停止。


## 2026-09-05 Phase 0.1 勘误（保留上述历史记录）

- Phase 0 的 JSON/log 有22个原件存在但未被Git跟踪，原因是根目录 `*.json`/`*.log`。Phase 0.1通过局部精确放行补档，原始内容及CSV未改写；清单见 `phase01_runs/archive_20260905T1238/archive_inventory.json`。
- 历史 `artifact_script_sha256_at_finalization` 只是归档时脚本hash，不能当作运行时hash；旧运行时脚本hash为 UNKNOWN。没有用当前环境补造旧manifest。
- 旧 `provider_replay.py` 从actual cache算attention reference，不能独立证明append正确；旧页表roll也不代表保留历史语义的迁移。旧PASS只保留当时实际检查范围，不升级为独立append通过。
- 旧 graph_batch 行复用了eager正确性，不能证明每个variant的Graph输出。历史CSV中的PASS不改写；新只读汇总把这类Graph行标为UNVERIFIED_LEGACY_GRAPH并排除排名。旧默认路径单独metadata检查也不能代表其他variant。
- 新Phase 0.1在独立run中使用调用前expected cache、负对照、每variant哨兵和Graph输出校验，并分别记录eager_correctness与graph_correctness。
- 本轮GPU UUID/PCI位置与Phase0不同，禁止跨轮性能加速比。详见Phase 0.1报告。
