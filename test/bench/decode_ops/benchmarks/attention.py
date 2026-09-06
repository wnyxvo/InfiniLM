"""Bounded attention path ablation including eager/Graph workload accounting."""
import argparse, hashlib, os, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, PAGE, reference, check_output, configure
from common.timing import measure_cuda, cpu_count_test

def h(*xs):
    chunks = []
    for x in xs:
        y = x.detach().contiguous()
        if y.dtype == torch.bfloat16: y = y.view(torch.uint16)
        chunks.append(y.cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(chunks)).hexdigest()

def set_mode(mode):
    # Clear all dispatch knobs so production_default observes source defaults.
    for key in list(os.environ):
        if key.startswith("INFINIOP_FLASH_"):
            os.environ.pop(key, None)
    if mode == "production_default":
        return
    if mode in ("auto", "capacity_strategy"):
        configure("default")
        os.environ["INFINIOP_FLASH_DECODE_SPLITKV"] = "auto"
        if mode == "capacity_strategy": os.environ["INFINIOP_FLASH_SPLITKV_STRATEGY"] = "capacity_v1"
        return
    configure("default")
    os.environ.update({"INFINIOP_FLASH_DECODE_KERNEL": "cta",
                       "INFINIOP_FLASH_GQA_FUSED": "0",
                       "INFINIOP_FLASH_DECODE_SPLITKV": "0" if mode in ("non_split_cta", "gqa_fused") else "1",
                       "INFINIOP_FLASH_NUM_SPLITS": "4"})
    if mode == "gqa_fused":
        os.environ["INFINIOP_FLASH_GQA_FUSED"] = "1"
    elif mode == "warp_splitkv":
        os.environ["INFINIOP_FLASH_DECODE_KERNEL"] = "warp"
    elif mode == "gqa_splitkv_2a":
        os.environ["INFINIOP_FLASH_GQA_SPLITKV"] = "1"
    elif mode in ("gqa_splitkv_shared", "gqa_splitkv_tile"):
        os.environ["INFINIOP_FLASH_GQA_SHARED_SPLITKV"] = "1"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--samples', type=int, default=5)
    p.add_argument('--graph-workloads', type=int, default=100); p.add_argument('--graph-replays', type=int, default=2); p.add_argument('--dtype', choices=('fp16','bf16'), default='fp16'); p.add_argument('--round', type=int, default=1)
    args = p.parse_args(); torch.cuda.set_device(0); run = Run(args.output); rows = []
    modes = ('production_default','auto','capacity_strategy','non_split_cta','gqa_fused','warp_splitkv','cta_splitkv','gqa_splitkv_2a','gqa_splitkv_shared','gqa_splitkv_tile')
    if args.round % 2 == 0: modes = tuple(reversed(modes))
    try:
        for mode in modes:
            for batch, length in ((1,256),(1,2049),(4,2049),(16,2049)):
                dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16; cap = (length + PAGE - 1) // PAGE
                torch.manual_seed(20260906 + batch + length)
                q = torch.randn(batch,32,128,device='cuda',dtype=dtype)
                k = torch.randn(batch*cap,8,PAGE,128,device='cuda',dtype=dtype); v = torch.randn_like(k)
                table = torch.randperm(batch*cap,device='cuda',dtype=torch.int32).reshape(batch,cap)
                lens = torch.full((batch,),length,device='cuda',dtype=torch.int32); out = torch.empty_like(q)
                native = Native(); call = native.operation('PagedAttention',[out,q,k,v,table,lens]); set_mode(mode)
                ref = reference(q,k,v,table,[length]*batch)
                out.fill_(float('nan')); call(); torch.cuda.synchronize(); eager_corr = check_output(out,ref)
                out.fill_(float('nan')); graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.graph_workloads): call()
                torch.cuda.synchronize(); out.fill_(float('nan')); graph.replay(); torch.cuda.synchronize(); graph_corr = check_output(out,ref)
                eager = measure_cuda(call,samples=args.samples,workload_calls=10,warmup=1)
                out.fill_(float('nan')); gm = measure_cuda(graph.replay,samples=args.samples,workload_calls=args.graph_replays,warmup=1)
                for sample in gm['samples']:
                    sample['normalized_us'] = sample['elapsed_ms']*1000/(args.graph_workloads*args.graph_replays)
                    sample['formula'] = 'elapsed_ms/(workloads_per_graph*graph_replays)'
                vals = [x['normalized_us'] for x in gm['samples']]
                split = mode in ('production_default','warp_splitkv','cta_splitkv','gqa_splitkv_2a','gqa_splitkv_shared','gqa_splitkv_tile')
                gm.update(median_us=statistics.median(vals), min_us=min(vals), max_us=max(vals),
                          workloads_per_graph=args.graph_workloads, graph_replays=args.graph_replays,
                          total_logical_workloads=args.graph_workloads*args.graph_replays,
                          kernel_calls_per_workload=2 if split else 1)
                rows.append({'mode':mode,'batch':batch,'length':length,'dtype':str(dtype),
                    'num_splits':4 if split else 0,'workspace_bytes':'descriptor-managed',
                    'eager_correctness':eager_corr['attention_correctness'],'graph_correctness':graph_corr['attention_correctness'],
                    'input_hash':h(q,k,v,table,lens),'eager':eager,'graph_batch':gm,
                    'result':'PASS' if eager_corr['attention_correctness']=='PASS' and graph_corr['attention_correctness']=='PASS' else 'FAIL'})
                del graph; native.close()
        run.add(rows=rows,cpu_count_test=cpu_count_test(),round=args.round,result='PASS' if all(r['result']=='PASS' for r in rows) else 'FAIL',
                graph_workloads=args.graph_workloads,graph_replays=args.graph_replays)
        return run.finish()
    except Exception as e:
        run.finish(e); raise
if __name__=='__main__': raise SystemExit(main())
