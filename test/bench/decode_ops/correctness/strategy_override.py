"""Minimal dispatch matrix for NUM_SPLITS/strategy precedence (Milestone 4)."""
import argparse, ctypes as C, os, re, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, capture_stderr

def env_for(case):
    for k in list(os.environ):
        if k.startswith('INFINIOP_FLASH_'): os.environ.pop(k, None)
    os.environ.update({
        'INFINIOP_FLASH_DECODE_KERNEL':'cta', 'INFINIOP_FLASH_GQA_FUSED':'1',
        'INFINIOP_FLASH_DECODE_SPLITKV':'auto',
        'INFINIOP_FLASH_SPLITKV_STRATEGY':'capacity_v1',
        'INFINIOP_FLASH_DEBUG_DISPATCH':'1', 'INFINIOP_FLASH_DEBUG_SPLITS':'1'})
    if case == 'numeric2': os.environ['INFINIOP_FLASH_NUM_SPLITS']='2'
    elif case == 'auto': os.environ['INFINIOP_FLASH_NUM_SPLITS']='auto'
    elif case == 'closed': os.environ['INFINIOP_FLASH_DECODE_SPLITKV']='0'
    elif case == 'invalid': os.environ['INFINIOP_FLASH_NUM_SPLITS']='bogus'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    torch.cuda.set_device(0); run=Run(a.output); native=Native(); rows=[]
    expected={
        'numeric2': {'path':'splitkv_cta','split':'1','num_splits':'2','strategy_reason':'numeric_override'},
        'auto': {'path':'splitkv_cta','split':'1','num_splits':'4','strategy_reason':'capacity_v1'},
        'unset': {'path':'splitkv_cta','split':'1','num_splits':'4','strategy_reason':'capacity_v1'},
        'closed': {'path':'cta_gqa_fused','split':'0','strategy_reason':'split_disabled'},
        'invalid': {'path':'splitkv_cta','split':'1','num_splits':'4','strategy_reason':'invalid_num_splits_default'},
        'out_of_scope': {'path':'splitkv_cta','split':'1','num_splits':'2','strategy_reason':'out_of_scope_fallback'},
    }
    try:
        for case in ('numeric2','auto','unset','closed','invalid','out_of_scope'):
            heads=16 if case=='out_of_scope' else 32; kv=8; batch=4; cap=9
            dtype=torch.float16
            q=torch.randn(batch,heads,128,device='cuda',dtype=dtype)
            k=torch.randn(batch*cap,kv,256,128,device='cuda',dtype=dtype); v=torch.randn_like(k)
            table=torch.arange(batch*cap,device='cuda',dtype=torch.int32).reshape(batch,cap)
            lens=torch.full((batch,),2049,device='cuda',dtype=torch.int32); out=torch.empty_like(q)
            call=native.operation('PagedAttention',[out,q,k,v,table,lens]); env_for(case)
            with capture_stderr() as f:
                call(); torch.cuda.synchronize(); C.CDLL(None).fflush(None); f.seek(0); trace=f.read().decode()
            lines=[x for x in trace.splitlines() if 'dispatch: path=' in x]
            if not lines: raise RuntimeError(f'no dispatch evidence for {case}: {trace}')
            dispatch=dict(re.findall(r'(\w+)=([^ ]+)',lines[-1].split('dispatch: ')[1]))
            exp=expected[case]
            if any(dispatch.get(k) != v for k,v in exp.items()):
                raise RuntimeError(f'{case} dispatch mismatch: expected {exp}, got {dispatch}')
            rows.append({'case':case,'heads':heads,'requested_num_splits':os.environ.get('INFINIOP_FLASH_NUM_SPLITS'),'expected':exp,'dispatch':dispatch,'trace':trace,'result':'PASS'})
        run.add(rows=rows, result='PASS'); return run.finish()
    except Exception as e:
        run.finish(e); raise
    finally: native.close()
if __name__=='__main__': raise SystemExit(main())
