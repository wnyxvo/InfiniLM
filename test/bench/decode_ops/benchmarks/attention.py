"""Bounded attention path ablation including batched Graph workload accounting."""
import argparse, os, torch, hashlib, statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, PAGE, reference, check_output, configure
from common.timing import measure_cuda, cpu_count_test

def h(*xs): return hashlib.sha256(''.join(str(x.detach().contiguous().cpu().numpy().tobytes()) for x in xs).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--samples',type=int,default=5); p.add_argument('--graph-workloads',type=int,default=100); p.add_argument('--graph-replays',type=int,default=2); args=p.parse_args(); torch.cuda.set_device(0); run=Run(args.output); rows=[]
 try:
  for mode in ('default','gqa_fused','splitkv','gqa_splitkv'):
   for batch,length in ((1,256),(1,2049),(4,2049),(16,2049)):
    dtype=torch.float16; cap=(length+PAGE-1)//PAGE; torch.manual_seed(20260906+batch+length); q=torch.randn(batch,32,128,device='cuda',dtype=dtype); k=torch.randn(batch*cap,8,PAGE,128,device='cuda',dtype=dtype); v=torch.randn_like(k); table=torch.randperm(batch*cap,device='cuda',dtype=torch.int32).reshape(batch,cap); lens=torch.full((batch,),length,device='cuda',dtype=torch.int32); out=torch.empty_like(q); native=Native(); call=native.operation('PagedAttention',[out,q,k,v,table,lens]); configure('default'); os.environ.update({'INFINIOP_FLASH_DECODE_KERNEL':'cta','INFINIOP_FLASH_GQA_FUSED':'0','INFINIOP_FLASH_DECODE_SPLITKV':'1','INFINIOP_FLASH_NUM_SPLITS':'4'}); os.environ.pop('INFINIOP_FLASH_GQA_SPLITKV',None); os.environ.pop('INFINIOP_FLASH_DEBUG_DISPATCH',None); os.environ.pop('INFINIOP_FLASH_DEBUG_SPLITS',None)
    if mode=='default': os.environ['INFINIOP_FLASH_DECODE_SPLITKV']='0'; os.environ['INFINIOP_FLASH_GQA_FUSED']='0'
    elif mode=='gqa_fused': os.environ['INFINIOP_FLASH_DECODE_SPLITKV']='0'; os.environ['INFINIOP_FLASH_GQA_FUSED']='1'
    elif mode=='gqa_splitkv': os.environ['INFINIOP_FLASH_GQA_SPLITKV']='1'
    out.fill_(float('nan')); call(); torch.cuda.synchronize(); corr=check_output(out,reference(q,k,v,table,[length]*batch));
    eager=measure_cuda(call,samples=args.samples,workload_calls=10,warmup=1)
    out.fill_(float('nan')); graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(args.graph_workloads): call()
    torch.cuda.synchronize(); out.fill_(float('nan')); gm=measure_cuda(graph.replay,samples=args.samples,workload_calls=args.graph_replays,warmup=1)
    # gm elapsed is per replay; normalize explicitly by workloads_per_graph * replays.
    for sample in gm['samples']: sample['normalized_us']=sample['elapsed_ms']*1000/(args.graph_workloads*args.graph_replays); sample['formula']='elapsed_ms/(workloads_per_graph*graph_replays)'
    vals=[x['normalized_us'] for x in gm['samples']]; gm.update(median_us=statistics.median(vals),min_us=min(vals),max_us=max(vals),workloads_per_graph=args.graph_workloads,graph_replays=args.graph_replays,total_logical_workloads=args.graph_workloads*args.graph_replays,kernel_calls_per_workload=1)
    rows.append({'mode':mode,'batch':batch,'length':length,'dtype':'torch.float16','num_splits':4 if mode in ('splitkv','gqa_splitkv') else 0,'workspace_bytes':'descriptor-managed','correctness':corr['attention_correctness'],'input_hash':h(q,k,v,table,lens),'eager':eager,'graph_batch':gm,'result':'PASS' if corr['attention_correctness']=='PASS' else 'FAIL'})
    del graph; native.close()
  run.add(rows=rows,cpu_count_test=cpu_count_test(),result='PASS' if all(r['result']=='PASS' for r in rows) else 'FAIL',graph_workloads=args.graph_workloads,graph_replays=args.graph_replays)
  return run.finish()
 except Exception as e: run.finish(e); raise
if __name__=='__main__': raise SystemExit(main())
