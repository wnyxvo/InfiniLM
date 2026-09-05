"""Bounded Phase 0 baseline; reuses InfiniCore's ctypes API registration.

No installation/build or production changes. CUDA events include enqueue gaps in
eager mode; graph_batch measures 100 captured calls, not the InfiniCore Graph API.
"""
import argparse
import ctypes as C
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

P = argparse.ArgumentParser()
P.add_argument('--core', type=Path, default=Path(__file__).resolve().parents[4] / 'InfiniCore')
P.add_argument('--output', type=Path, required=True)
P.add_argument('--quick', action='store_true')
P.add_argument('--fp16', action='store_true')
P.add_argument('--caching-only', action='store_true')
P.add_argument('--aligned', action='store_true')
P.add_argument('--variant', choices=['default','auto','warp','cta','gqa','split2','split8'])
args = P.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.core / 'test/infiniop'))
import torch
from libinfiniop import LIBINFINIOP as lib, check_error, infiniopTensorDescriptor_t, infiniopOperatorDescriptor_t, infiniopHandle_t

torch.manual_seed(20260905)
torch.cuda.set_device(0)
check_error(lib.infinirtSetDevice(1, 0))
handle = infiniopHandle_t()
check_error(lib.infiniopCreateHandle(C.byref(handle)))
run_id = args.output.name
knobs = ['DECODE_KERNEL', 'GQA_FUSED', 'CTA_TILE', 'CTA_THREADS', 'DECODE_SPLITKV', 'NUM_SPLITS', 'DEBUG_DISPATCH', 'DEBUG_SPLITS']
initial_env = {k: v for k, v in os.environ.items() if k.startswith(('INFINIOP_FLASH_', 'INFINICORE_GRAPH', 'INFINICORE_DISABLE_DEVICE_GRAPH')) or k in ['CUDA_VISIBLE_DEVICES','INFINI_ROOT','LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CUDA_HOME']}
variants = {'default': {}, 'auto': {'DECODE_SPLITKV':'auto'}, 'warp': {'DECODE_KERNEL':'warp','DECODE_SPLITKV':'0','GQA_FUSED':'0'}, 'cta': {'DECODE_SPLITKV':'0','GQA_FUSED':'0'}, 'gqa': {'DECODE_SPLITKV':'0'}, 'split2': {'NUM_SPLITS':'2'}, 'split8': {'NUM_SPLITS':'8'}}

if args.variant: variants = {args.variant: variants[args.variant]}

def command(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

manifest = {'run_id':run_id,'command':sys.argv,'seed':20260905,'environment':initial_env,'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_path':torch.__file__,'cuda':torch.version.cuda,'gpu':str(torch.cuda.get_device_properties(0)), 'repos':{}, 'protocol':{'warmup':10,'samples':7,'repeats':100,'cache':'repeated hot KV; no rotating KV','eager':'CUDA events on supplied current stream; includes CPU enqueue gaps','graph_batch':'100 native calls captured in torch CUDAGraph, 7 replay samples; not framework Graph','attention_tolerance':'BF16 atol=.005 rtol=.05; F16 atol=.001 rtol=.01; independent FP32 softmax/reference','scope':'direct native C API, no C++ provider'}}
for repo in [Path(__file__).resolve().parents[3], args.core]:
    manifest['repos'][str(repo)] = {k:command(['git','-C',str(repo),*v]) for k,v in {'sha':['rev-parse','HEAD'],'branch':['branch','--show-current'],'dirty':['status','--short'],'submodules':['submodule','status']}.items()}
manifest['build_config'] = (args.core / '.xmake/linux/x86_64/xmake.conf').read_text()
manifest['tools'] = [command(c) for c in [['nvidia-smi'],['nvcc','--version'],['g++','--version'],['nsys','--version'],['ncu','--version'],['compute-sanitizer','--version']]]
manifest['libraries'] = {}
for p in [Path(lib.libop._name), Path(lib.librt._name), args.core/'build/linux/x86_64/release/libinfiniop.so']:
    manifest['libraries'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
manifest['loaded_libraries'] = sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if '/libinfini' in l or '/libflash-attn' in l})
models = list((Path.home()/'huggingface').rglob('config.json'))
manifest['models'] = {str(p):json.loads(p.read_text()) for p in models}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

def desc(t):
    d = infiniopTensorDescriptor_t()
    dtype = {torch.bfloat16:19,torch.float16:12,torch.float32:13,torch.int32:5,torch.int64:6}[t.dtype]
    check_error(lib.infiniopCreateTensorDescriptor(C.byref(d),t.ndim,(C.c_uint64*t.ndim)(*t.shape),(C.c_int64*t.ndim)(*t.stride()),dtype))
    return d

def operation(name, tensors):
    ds = [desc(t) for t in tensors]
    op = infiniopOperatorDescriptor_t()
    extras = [None, 128**-.5] if name == 'PagedAttention' else []
    check_error(getattr(lib,'infiniopCreate'+name+'Descriptor')(handle,C.byref(op),*ds,*extras))
    n = C.c_uint64()
    check_error(getattr(lib,'infiniopGet'+name+'WorkspaceSize')(op,C.byref(n)))
    ws = torch.empty(max(n.value,1),device='cuda',dtype=torch.uint8)
    ptrs = [t.data_ptr() for t in tensors]
    def call():
        stream = torch.cuda.current_stream().cuda_stream
        check_error(getattr(lib,'infiniop'+name)(op,ws.data_ptr(),n.value,*ptrs,*([None] if name=='PagedAttention' else []),stream))
    def close():
        check_error(getattr(lib,'infiniopDestroy'+name+'Descriptor')(op))
        for d in ds: check_error(lib.infiniopDestroyTensorDescriptor(d))
    return call, close, n.value, ws

def reference(q,k,v,bt,lengths):
    outs=[]
    for b,L in enumerate(lengths):
        ids=bt[b,:((L+255)//256)].long()
        kk=k[ids].permute(0,2,1,3).reshape(-1,k.shape[1],128)[:L].float().repeat_interleave(q.shape[1]//k.shape[1],1)
        vv=v[ids].permute(0,2,1,3).reshape(-1,v.shape[1],128)[:L].float().repeat_interleave(q.shape[1]//k.shape[1],1)
        scores=torch.einsum('hd,lhd->hl',q[b].float(),kk)*128**-.5
        outs.append(torch.einsum('hl,lhd->hd',scores.softmax(-1),vv))
    return torch.stack(outs)

fields=['run_id','op','case','B','lengths','capacity_pages','dtype','requested_variant','requested_env','actual_provider','actual_kernel','requested_graph_mode','actual_execution_mode','correctness','max_abs_error','workspace_bytes','shapes','strides','sample','repeats','latency_us']
f=(args.output/'raw.csv').open('w'); writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()

def measure(call,row):
    for _ in range(10): call()
    torch.cuda.synchronize()
    for mode in ['eager','graph_batch']:
        if mode=='graph_batch':
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):call()
            run=graph.replay
        else:
            def run():
                for _ in range(100):call()
        samples=[]
        for i in range(7):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();run();end.record();end.synchronize()
            us=start.elapsed_time(end)*10
            samples.append(us)
            writer.writerow(dict(row,requested_graph_mode=mode,actual_execution_mode=mode,sample=i,repeats=100,latency_us=us));f.flush()
        print(row['case'],row['requested_variant'],mode,statistics.median(samples),flush=True)

def rowbase(name,case,tensors):
    return dict(run_id=run_id,op=name,case=case,dtype=str(tensors[0].dtype),actual_provider='native_C_API',actual_kernel='see dispatch.log; caching symbol confirmed separately',shapes=json.dumps([list(t.shape) for t in tensors]),strides=json.dumps([list(t.stride()) for t in tensors]))

cases=[('b1l256',[256],1),('b1l8192',[8192],32),('b16l2048',[2048]*16,8),('b32l8192',[8192]*32,32),('capacity_wide',[256],32),('length_short',[17],32),('mixed',[17,255,256,2049],32)]
if args.quick:cases=cases[:1]
if args.caching_only:cases=[]
for case,lengths,capacity in cases:
    B=len(lengths);H=32;K=8
    torch.manual_seed(20260905)
    q=torch.randn(B,H,128,device='cuda',dtype=torch.float16 if args.fp16 else torch.bfloat16)
    k=torch.randn(B*capacity,K,256,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    bt=torch.randperm(B*capacity,device='cuda',dtype=torch.int32).reshape(B,capacity)
    lens=torch.tensor(lengths,device='cuda',dtype=torch.int32);out=torch.empty_like(q)
    ref=reference(q,k,v,bt,lengths)
    tensors=[out,q,k,v,bt,lens]
    call,close,n,ws=operation('PagedAttention',tensors)
    for variant,config in variants.items():
        for knob in knobs:os.environ.pop('INFINIOP_FLASH_'+knob,None)
        for knob,value in config.items():os.environ['INFINIOP_FLASH_'+knob]=value
        os.environ['INFINIOP_FLASH_DEBUG_DISPATCH']='1'
        os.environ['INFINIOP_FLASH_DEBUG_SPLITS']='1'
        print('TRACE',case,variant,flush=True);call();torch.cuda.synchronize()
        os.environ.pop('INFINIOP_FLASH_DEBUG_DISPATCH');os.environ.pop('INFINIOP_FLASH_DEBUG_SPLITS')
        error=(out.float()-ref).abs().max().item()
        passed=torch.allclose(out.float(),ref,atol=.001 if args.fp16 else .005,rtol=.01 if args.fp16 else .05)
        row=dict(rowbase('attention',case,tensors),B=B,lengths=json.dumps(lengths),capacity_pages=capacity,requested_variant=variant,requested_env=json.dumps(config),correctness='PASS' if passed else 'FAIL',max_abs_error=error,workspace_bytes=n)
        if not passed:
            row['actual_execution_mode']='eager_correctness_only'
            writer.writerow(row);f.flush()
            print('FAIL',case,variant,'per_head_max',(out.float()-ref).abs().amax(dim=(0,2)).tolist(),flush=True)
            continue
        measure(call,row)
    # Same captured graph, in-place metadata updates and multiple replay checks.
    for knob in knobs:os.environ.pop('INFINIOP_FLASH_'+knob,None)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for step in range(3):
        updated=[max(1,L-step-1) for L in lengths]
        lens.copy_(torch.tensor(updated,device='cuda',dtype=torch.int32))
        bt.copy_(bt.roll(1,1))
        graph.replay();torch.cuda.synchronize()
        expected=reference(q,k,v,bt,updated)
        assert torch.allclose(out.float(),expected,atol=.005,rtol=.05)
    print('metadata replay PASS',case,flush=True)
    del graph
    close()

for dtype in [torch.bfloat16,torch.float16,torch.float32]:
    for B in ([1] if args.quick else [1,4,16,32]):
        for dv in ([128] if args.quick else [128,96]):
            # Offset by one element, padded token/head strides, padded V cache.
            k=torch.randn(B,8,129,device='cuda',dtype=dtype)[...,1:]
            v=torch.randn(B,8,dv+1,device='cuda',dtype=dtype)[...,1:]
            kc=torch.randn(2,8,256,129,device='cuda',dtype=dtype)[...,1:]
            vc=torch.randn_like(kc)
            if args.aligned:
                k=k.contiguous();v=v.contiguous();kc=kc.contiguous();vc=vc.contiguous()
            slots=torch.arange(B,device='cuda',dtype=torch.int64)+255
            if B>1:slots[-1]=-1
            expected_k=kc.clone();expected_v=vc.clone()
            for i,s in enumerate(slots.tolist()):
                if s>=0:expected_k[s//256,:,s%256,:]=k[i];expected_v[s//256,:,s%256,:dv]=v[i]
            tensors=[kc,vc,k,v,slots]
            call,close,n,ws=operation('PagedCaching',tensors);call();torch.cuda.synchronize()
            assert torch.equal(kc,expected_k) and torch.equal(vc,expected_v)
            row=dict(rowbase('caching',f't{B}dv{dv}_aligned{args.aligned}',tensors),B=B,requested_variant='default',requested_env='{}',correctness='PASS_bitwise_all_cache',workspace_bytes=n)
            measure(call,row);close()
f.close();check_error(lib.infiniopDestroyHandle(handle))
