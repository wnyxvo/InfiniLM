"""Bounded repeatability probe for the Split4 cross-check anomaly; no timing."""
import argparse
import sys
import torch
from correctness_support import Run, Native, SEED, PAGE, configure, reference, check_output, trace_call


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variant',choices=['default','gqa'],default='default');p.add_argument('--repeats',type=int,default=30);a=p.parse_args()
    torch.manual_seed(SEED);run=Run(a.output)
    try:
        lengths=[1,7,8,9,255,256,257,2049]*2;batch=16;capacity=9
        q=torch.randn(batch,32,128,device='cuda',dtype=torch.float16)
        k=torch.randn(batch*capacity,8,PAGE,128,device='cuda',dtype=q.dtype)
        v=torch.randn_like(k)
        table=torch.randperm(batch*capacity,device='cuda',dtype=torch.int32).reshape(batch,capacity)
        lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q);native=Native()
        call=native.operation('PagedAttention',[out,q,k,v,table,lens])
        ref=reference(q,k,v,table,lengths)
        config,dispatch,trace=trace_call(call,a.variant)
        (run.path/'dispatch.log').write_text(trace)
        for repeat in range(a.repeats):
            out.fill_(float('nan'));call();torch.cuda.synchronize()
            result=check_output(out,ref)
            run.add(dtype=str(q.dtype),variant=a.variant,mode='eager',repeat=repeat,
                    actual_dispatch=dispatch,requested_config=config,
                    result=result['attention_correctness'],**result)
        native.close();return run.finish()
    except Exception as error:
        run.finish(error);raise


if __name__=='__main__':sys.exit(main())
