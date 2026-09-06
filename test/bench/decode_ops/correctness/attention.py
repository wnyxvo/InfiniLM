"""Bounded GQA regression with separate eager and per-variant CUDA Graph checks."""
import argparse
import json
import sys

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import (Run, Native, SEED, PAGE, configure, trace_call,
                                 reference, check_output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--minimal', action='store_true')
    p.add_argument('--case', help='Run one named case for bounded failure diagnosis')
    args = p.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    run = Run(args.output)
    try:
        cases = [(f'l{n}', [n]) for n in [1,7,8,9,255,256,257,2049,8192]]
        cases += [('mixed4',[7,9,255,257]), ('mixed16',[1,7,8,9,255,256,257,2049]*2)]
        if args.minimal:
            cases = [('l256',[256])]
        if args.case:
            cases=[c for c in cases if c[0]==args.case]
            if not cases:raise ValueError('Unknown case: '+args.case)
        for dtype in [torch.float16,torch.bfloat16]:
            for case, lengths in cases:
                batch = len(lengths)
                capacity = (max(lengths)+PAGE-1)//PAGE
                torch.manual_seed(SEED)
                q = torch.randn(batch,32,128,device='cuda',dtype=dtype)
                k = torch.randn(batch*capacity,8,PAGE,128,device='cuda',dtype=dtype)
                v = torch.randn_like(k)
                table = torch.randperm(batch*capacity,device='cuda',dtype=torch.int32).reshape(batch,capacity)
                lens = torch.tensor(lengths,device='cuda',dtype=torch.int32)
                out = torch.full_like(q,float('nan'))
                native = Native()
                call = native.operation('PagedAttention',[out,q,k,v,table,lens])
                original_q = q.clone()
                variants = ['gqa']
                if case in ('l1','l256','mixed4','mixed16'):
                    variants += ['auto']
                if case in ('l1','l256','mixed4','l8192'):
                    variants += ['default','cta','split2']
                for variant in variants:
                    q.copy_(original_q)
                    lens.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                    torch.cuda.synchronize()
                    # Independent known native Split4 cross-check on same inputs.
                    configure('default')
                    out.fill_(float('nan')); call(); torch.cuda.synchronize()
                    ref = reference(q,k,v,table,lengths)
                    cross = check_output(out,ref)
                    control = out.clone()
                    config, dispatch, trace = trace_call(call,variant)
                    with (run.path/'dispatch.log').open('a') as f:
                        f.write(f'TRACE {dtype} {case} {variant}\n{trace}\n')
                    expected_path = {'gqa':'cta_gqa_fused','cta':'cta_nosplit','default':'splitkv_cta','split2':'splitkv_cta','auto':'cta_gqa_fused'}[variant]
                    if dispatch.get('path') != expected_path:
                        raise AssertionError(f'requested {variant}, expected {expected_path}, observed {dispatch}')
                    common = dict(case=case,dtype=str(dtype),variant=variant,
                                  requested_config=config,actual_dispatch=dispatch,
                                  provider='native_C_API',shape=list(q.shape),
                                  capacity_pages=capacity,scale=128**-.5,
                                  crosscheck_split4=cross['attention_correctness'],crosscheck_metrics=cross)
                    out.fill_(float('nan')); call(); torch.cuda.synchronize()
                    result = check_output(out,ref)
                    delta = (out.float()-control.float()).abs().amax(dim=(0,2)).tolist()
                    run.add(**common,mode='eager',lengths=lengths,
                            eager_correctness=result['attention_correctness'],graph_correctness='NOT_RUN',
                            result='PASS' if result['attention_correctness']=='PASS' and cross['attention_correctness']=='PASS' else 'FAIL',
                            crosscheck_per_head_max=[x if x<float('inf') else 'NONFINITE' for x in delta],**result)
                    # A fresh graph per requested variant. Never reuse eager PASS.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call()
                    for step in range(3):
                        # Append-free attention test: reveal preinitialized cache
                        # tokens up to capacity; provider_replay tests real append.
                        updated = [min(capacity*PAGE,n+step) for n in lengths]
                        lens.copy_(torch.tensor(updated,device='cuda',dtype=torch.int32))
                        q.normal_()
                        out.fill_(float('nan'))
                        torch.cuda.synchronize()
                        graph.replay(); torch.cuda.synchronize()
                        result = check_output(out,reference(q,k,v,table,updated))
                        run.add(**common,mode='torch_cuda_graph',step=step,lengths=updated,
                                eager_correctness='NOT_RUN',graph_correctness=result['attention_correctness'],
                                result=result['attention_correctness'],**result)
                    del graph
                    print(dtype,case,variant,'recorded',flush=True)
                native.close()
        return run.finish()
    except Exception as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    sys.exit(main())
