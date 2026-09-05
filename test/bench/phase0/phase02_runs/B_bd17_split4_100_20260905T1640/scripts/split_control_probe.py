"""Bounded repeatability probe for the Split4 cross-check anomaly; no timing."""
import argparse
import sys
import hashlib
import torch
from correctness_support import Run, Native, SEED, PAGE, configure, reference, check_output, trace_call


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variant',choices=['default','gqa'],default='default');p.add_argument('--repeats',type=int,default=30);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--lengths',default='1,7,8,9,255,256,257,2049,1,7,8,9,255,256,257,2049');a=p.parse_args()
    torch.manual_seed(a.seed);run=Run(a.output)
    try:
        lengths=[int(x) for x in a.lengths.split(',')];batch=len(lengths);capacity=max(9, (max(lengths)+PAGE-1)//PAGE)
        q=torch.randn(batch,32,128,device='cuda',dtype=torch.float16)
        k=torch.randn(batch*capacity,8,PAGE,128,device='cuda',dtype=q.dtype)
        v=torch.randn_like(k)
        table=torch.randperm(batch*capacity,device='cuda',dtype=torch.int32).reshape(batch,capacity)
        lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q);native=Native()
        call=native.operation('PagedAttention',[out,q,k,v,table,lens])
        ref=reference(q,k,v,table,lengths)
        def thash(x):
            return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        input_hash=hashlib.sha256(''.join(thash(x) for x in (q,k,v,table,lens)).encode()).hexdigest()
        reference_hash=thash(ref)
        config,dispatch,trace=trace_call(call,a.variant)
        (run.path/'dispatch.log').write_text(trace)
        for repeat in range(a.repeats):
            out.fill_(float('nan'));call();torch.cuda.synchronize()
            result=check_output(out,ref)
            diff=(out.float()-ref.float()).abs()
            threshold=0.001+0.01*ref.float().abs()
            bad=torch.nonzero(diff > threshold, as_tuple=False)
            coords=[]
            for coord in bad[:32].tolist():
                b,h,d=coord; coords.append({'batch':b,'head':h,'dim':d,'actual':float(out[b,h,d]),'expected':float(ref[b,h,d]),'abs_error':float(diff[b,h,d]),'threshold':float(threshold[b,h,d])})
            run.add(dtype=str(q.dtype),variant=a.variant,mode='eager',repeat=repeat,
                    actual_dispatch=dispatch,requested_config=config,
                    input_hash=input_hash,reference_hash=reference_hash,output_hash=thash(out),
                    failed_coords=coords,inputs_unchanged=(input_hash==hashlib.sha256(''.join(thash(x) for x in (q,k,v,table,lens)).encode()).hexdigest()),
                    reference_unchanged=(reference_hash==thash(ref)),
                    result=result['attention_correctness'],**result)
        native.close();return run.finish()
    except Exception as error:
        run.finish(error);raise


if __name__=='__main__':sys.exit(main())
