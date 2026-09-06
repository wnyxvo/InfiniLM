"""Independent append verification for eager and InfiniCore device Graph.

Expected caches are cloned BEFORE execution; attention never reads the actual
post-call cache as its reference. Negative controls are test-only injections.
"""
import argparse
import sys

if '--help' in sys.argv:
    _p=argparse.ArgumentParser(description='Independent append and InfiniCore Graph verification')
    _p.add_argument('--output');_p.add_argument('--flash',action='store_true');_p.add_argument('--variant',choices=['gqa','auto','default','cta','split2','strategy'])
    _p.parse_args();sys.exit(0)

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import (Run, SEED, PAGE, configure, trace_call,
                                 reference, check_output, append_expected)
import infinicore as ic
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.mha_kvcache import mha_kvcache


def wrap(t):
    types = {torch.float16:ic.float16,torch.bfloat16:ic.bfloat16,
             torch.int32:ic.int32,torch.int64:ic.int64}
    return ic.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),
                               dtype=types[t.dtype],device=ic.device('cuda',0))


def check_caches(k,v,ek,ev):
    # All finite initialized elements: bitwise equality of whole cache also
    # proves that unwritten positions, including old relocated pages, persist.
    return torch.equal(k.view(torch.int16),ek.contiguous().view(torch.int16)) and torch.equal(v.view(torch.int16),ev.contiguous().view(torch.int16))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,help='New run directory (must not exist)')
    p.add_argument('--flash',action='store_true')
    p.add_argument('--variant',choices=['gqa','auto','default','cta','split2','strategy'])
    args=p.parse_args()
    torch.cuda.set_device(0)
    ic.set_device(ic.device('cuda',0))
    import os
    if 'INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS' in os.environ:
        raise RuntimeError('Device graph disabled: cannot verify requested mode')
    os.environ['INFINICORE_GRAPH_DEBUG']='1'
    run=Run(args.output)
    try:
        variants=[args.variant] if args.variant else (['default'] if args.flash else ['gqa','auto','default','cta','split2'])
        for dtype in [torch.float16,torch.bfloat16]:
            for variant in variants:
                configure(variant)
                if variant == 'strategy': os.environ['INFINIOP_FLASH_SPLITKV_STRATEGY'] = 'capacity_v1'
                torch.manual_seed(SEED)
                batch=4; pages_per_seq=2
                # Extra spare page per sequence allows a semantic relocation:
                # copy historical page contents, then change only its mapping.
                shape=(batch*pages_per_seq+batch,8,PAGE,128)
                initial_k=torch.randn(shape,device='cuda',dtype=dtype)
                initial_v=torch.randn_like(initial_k)
                initial_table=torch.randperm(batch*pages_per_seq).reshape(batch,pages_per_seq)
                q=torch.randn(batch,32,128,device='cuda',dtype=dtype)
                out=torch.empty_like(q)
                newk=torch.randn(batch,8,128,device='cuda',dtype=dtype)
                newv=torch.randn_like(newk)
                if args.flash:
                    kstore=initial_k.permute(0,2,1,3).contiguous()
                    vstore=initial_v.permute(0,2,1,3).contiguous()
                    k=kstore.permute(0,2,1,3);v=vstore.permute(0,2,1,3)
                else:
                    k=initial_k.clone();v=initial_v.clone();kstore=k;vstore=v
                table=initial_table.to(device='cuda',dtype=torch.int32)
                lens=torch.tensor([255,256,257,8],device='cuda',dtype=torch.int32)
                slots=torch.tensor([int(initial_table[b,(n-1)//PAGE])*PAGE+(n-1)%PAGE for b,n in enumerate([255,256,257,8])],device='cuda',dtype=torch.int64)
                iq,ik,iv,ink,inv,ibt,ilens,islots,io=[wrap(t) for t in [q,k,v,newk,newv,table,lens,slots,out]]
                fq,fk,fv,fo=[wrap(t) for t in [q.unsqueeze(1),kstore,vstore,out.unsqueeze(1)]]
                def attention():
                    if args.flash:
                        mha_kvcache(fq,fk,fv,ilens,ibt,scale=128**-.5,out=fo)
                    else:
                        paged_attention(iq,ik,iv,ibt,ilens,scale=128**-.5,out=io)
                def ops():
                    paged_caching(ik,iv,ink,inv,islots)
                    attention()
                torch.cuda.synchronize()
                if not args.flash:
                    config,dispatch,trace=trace_call(attention,variant)
                    with (run.path/'dispatch.log').open('a') as f:
                        f.write(f'TRACE {dtype} {variant}\n{trace}\n')
                    want={'gqa':'cta_gqa_fused','auto':'cta_gqa_fused','cta':'cta_nosplit','default':'splitkv_cta','split2':'splitkv_cta'}
                    if variant != 'strategy' and dispatch.get('path')!=want[variant]: raise AssertionError(dispatch)
                else:
                    config=configure(variant);dispatch={'path':'builtin_FlashAttention; confirm with Nsight'}
                # Capture may run ops during warmup. Reset ACTUAL and expected
                # from the original pre-call snapshot after capture below.
                ic.start_graph_recording();ops();graph=ic.stop_graph_recording()
                for mode,injection in [('eager',None),('infinicore_graph',None),
                                       ('eager','skip_append'),('infinicore_graph','corrupt_written')]:
                    k.copy_(initial_k);v.copy_(initial_v)
                    ek=initial_k.clone();ev=initial_v.clone()
                    host_table=initial_table.clone()
                    detected=False
                    for step in range(3):
                        lengths=[255+step,256+step,257+step,8+step]
                        if step==1:
                            # Move each sequence's logical first page into an
                            # unused physical page, preserving history exactly.
                            for b in range(batch):
                                old=int(host_table[b,0]);new=batch*pages_per_seq+b
                                k[new].copy_(k[old]);v[new].copy_(v[old])
                                ek[new].copy_(ek[old]);ev[new].copy_(ev[old])
                                host_table[b,0]=new
                        host_slots=[int(host_table[b,(n-1)//PAGE])*PAGE+(n-1)%PAGE for b,n in enumerate(lengths)]
                        table.copy_(host_table.to(device='cuda',dtype=torch.int32))
                        lens.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                        slots.copy_(torch.tensor(host_slots,device='cuda',dtype=torch.int64))
                        q.normal_();newk.normal_();newv.normal_()
                        append_expected(ek,ev,newk,newv,host_slots)
                        expected=reference(q,ek,ev,table,lengths)
                        out.fill_(float('nan'));torch.cuda.synchronize()
                        if injection=='skip_append' and step==0:attention()
                        elif mode=='eager':ops()
                        else:graph.run()
                        ic.sync_stream();torch.cuda.synchronize()
                        if injection=='corrupt_written' and step==0:
                            s=host_slots[0];k[s//PAGE,0,s%PAGE,0]+=1
                        cache_pass=check_caches(k,v,ek,ev)
                        result=check_output(out,expected)
                        passed=cache_pass and result['attention_correctness']=='PASS'
                        if injection:
                            detected=not cache_pass
                            run.add(dtype=str(dtype),variant=variant,mode=mode,step=step,
                                    negative_control=injection,detected=detected,
                                    cache_correctness='PASS' if cache_pass else 'FAIL',
                                    result='PASS' if detected else 'FAIL',**result)
                            break
                        run.add(dtype=str(dtype),variant=variant,mode=mode,step=step,
                                requested_config=config,actual_dispatch=dispatch,
                                provider='native_caching+builtin_flash' if args.flash else 'native_C++',
                                lengths=lengths,block_table=host_table.tolist(),slots=host_slots,
                                cache_correctness='PASS' if cache_pass else 'FAIL',
                                eager_correctness=('PASS' if passed else 'FAIL') if mode=='eager' else 'NOT_RUN',
                                graph_correctness=('PASS' if passed else 'FAIL') if mode=='infinicore_graph' else 'NOT_RUN',
                                result='PASS' if passed else 'FAIL',**result)
                    print(dtype,variant,mode,injection or 'positive','recorded',flush=True)
                del graph
        return run.finish()
    except Exception as error:
        run.finish(error)
        raise


if __name__=='__main__':sys.exit(main())
