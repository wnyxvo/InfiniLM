"""Bounded correctness for the experimental GQA-aware split-KV path."""
import argparse, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, PAGE, reference, check_output, configure

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args(); torch.cuda.set_device(0); run=Run(args.output)
 try:
  for dtype in (torch.float16, torch.bfloat16):
   for lengths in ([1],[7],[255],[256],[257],[2049],[7,256,2049]):
    batch=len(lengths); cap=(max(lengths)+PAGE-1)//PAGE; torch.manual_seed(20260906)
    q=torch.randn(batch,32,128,device='cuda',dtype=dtype); k=torch.randn(batch*cap,8,PAGE,128,device='cuda',dtype=dtype); v=torch.randn_like(k)
    table=torch.randperm(batch*cap,device='cuda',dtype=torch.int32).reshape(batch,cap); lens=torch.tensor(lengths,device='cuda',dtype=torch.int32); out=torch.full_like(q,float('nan')); n=Native(); call=n.operation('PagedAttention',[out,q,k,v,table,lens])
    configure('default'); os.environ.update({'INFINIOP_FLASH_DECODE_KERNEL':'cta','INFINIOP_FLASH_DECODE_SPLITKV':'1','INFINIOP_FLASH_NUM_SPLITS':'4','INFINIOP_FLASH_GQA_SPLITKV':'1','INFINIOP_FLASH_GQA_FUSED':'0','INFINIOP_FLASH_DEBUG_DISPATCH':'1'})
    out.fill_(float('nan')); call(); torch.cuda.synchronize(); ref=reference(q,k,v,table,lengths); eager=check_output(out,ref)
    base_q=q.clone(); graph=torch.cuda.CUDAGraph(); out.fill_(float('nan'))
    with torch.cuda.graph(graph): call()
    q.copy_(base_q); lens.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32)); out.fill_(float('nan')); graph.replay(); torch.cuda.synchronize(); graph_ok=check_output(out,reference(q,k,v,table,lengths))
    run.add(dtype=str(dtype),lengths=lengths,batch=batch,capacity_pages=cap,num_splits=4,dispatch='splitkv_gqa_4warp',eager_correctness=eager['attention_correctness'],graph_correctness=graph_ok['attention_correctness'],result='PASS' if eager['attention_correctness']=='PASS' and graph_ok['attention_correctness']=='PASS' else 'FAIL',eager_metrics=eager,graph_metrics=graph_ok)
    del graph; n.close()
  return run.finish()
 except Exception as e: run.finish(e); raise
if __name__=='__main__': raise SystemExit(main())
