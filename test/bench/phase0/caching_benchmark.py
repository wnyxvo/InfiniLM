"""CUDA-event benchmark for paged_caching CTA thread candidates and append->attention."""
import argparse, ctypes as C, hashlib, json, os, statistics, sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',default='default');p.add_argument('--dtype',choices=['fp16','bf16'],default='fp16');p.add_argument('--repeats',type=int,default=20);p.add_argument('--calls',type=int,default=100);a=p.parse_args()
root=Path(__file__).resolve().parents[4];sys.path.insert(0,str(root/'InfiniCore/test/infiniop'))
from libinfiniop import LIBINFINIOP as lib,check_error,infiniopTensorDescriptor_t,infiniopOperatorDescriptor_t,infiniopHandle_t
outdir=Path(a.output);outdir.mkdir(parents=True,exist_ok=False);torch.cuda.set_device(0);check_error(lib.infinirtSetDevice(1,0));handle=infiniopHandle_t();check_error(lib.infiniopCreateHandle(C.byref(handle)))
dtype=torch.float16 if a.dtype=='fp16' else torch.bfloat16; dtype_id=12 if a.dtype=='fp16' else 19
threads={'default':None,'128':'128','256':'256'}[a.threads];
if threads is None: os.environ.pop('INFINIOP_PAGED_CACHING_THREADS',None)
else: os.environ['INFINIOP_PAGED_CACHING_THREADS']=threads

def desc(t):
 d=infiniopTensorDescriptor_t();check_error(lib.infiniopCreateTensorDescriptor(C.byref(d),t.ndim,(C.c_uint64*t.ndim)(*t.shape),(C.c_int64*t.ndim)(*t.stride()),{torch.float16:12,torch.bfloat16:19,torch.float32:13,torch.int32:5,torch.int64:6}[t.dtype]));return d
def makeop(name,ts):
 ds=[desc(t) for t in ts];op=infiniopOperatorDescriptor_t();extra=[None,128**-.5] if name=='PagedAttention' else []
 check_error(getattr(lib,'infiniopCreate'+name+'Descriptor')(handle,C.byref(op),*ds,*extra));n=C.c_uint64();check_error(getattr(lib,'infiniopGet'+name+'WorkspaceSize')(op,C.byref(n)));ws=torch.empty(max(1,n.value),device='cuda',dtype=torch.uint8)
 def call(): check_error(getattr(lib,'infiniop'+name)(op,ws.data_ptr(),n.value,*[t.data_ptr() for t in ts],*([None] if name=='PagedAttention' else []),torch.cuda.current_stream().cuda_stream))
 def close(): check_error(getattr(lib,'infiniopDestroy'+name+'Descriptor')(op));[lib.infiniopDestroyTensorDescriptor(d) for d in ds]
 return call,close

def thash(t):
 raw=t.detach().contiguous()
 if raw.dtype==torch.bfloat16: raw=raw.view(torch.uint16)
 return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()
def event_measure(run):
 vals=[]
 for _ in range(a.repeats):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();run();e.record();e.synchronize();vals.append(s.elapsed_time(e)*1000/a.calls)
 return {'median_us':statistics.median(vals),'min_us':min(vals),'max_us':max(vals),'samples_us':vals}
rows=[]
for ntok in [1,4,16,32]:
 k=torch.randn(ntok,8,128,device='cuda',dtype=dtype);v=torch.randn_like(k);pages=max(1,(ntok+255)//256);kc=torch.randn(pages,8,256,128,device='cuda',dtype=dtype);vc=torch.randn_like(kc);slots=torch.arange(ntok,device='cuda',dtype=torch.int64);ek=kc.clone();ev=vc.clone();
 for i,slot in enumerate(slots.tolist()):
  ek[slot//256,:,slot%256,:]=k[i]; ev[slot//256,:,slot%256,:]=v[i]
 call,close=makeop('PagedCaching',[kc,vc,k,v,slots]);call();torch.cuda.synchronize();correct=bool(torch.equal(kc,ek) and torch.equal(vc,ev));
 for _ in range(10): call()
 torch.cuda.synchronize()
 def reset(): kc.copy_(kc); vc.copy_(vc)
 eager=event_measure(lambda:[call() for _ in range(a.calls)])
 graph=torch.cuda.CUDAGraph();torch.cuda.synchronize()
 with torch.cuda.graph(graph):
  for _ in range(a.calls):call()
 torch.cuda.synchronize();graphm=event_measure(graph.replay)
 rows.append({'operation':'paged_caching','dtype':a.dtype,'threads':a.threads,'tokens':ntok,'correctness':'PASS' if correct else 'FAIL','cache_hash':thash(kc),'eager':eager,'graph':graphm,'calls_per_sample':a.calls});close();del graph
# append -> attention integration, same cache and lengths, measured as two-op sequence
B=4;lengths=[1,4,16,32];ntok=sum(lengths);q=torch.randn(B,32,128,device='cuda',dtype=dtype);k=torch.randn(ntok,8,128,device='cuda',dtype=dtype);v=torch.randn_like(k);slots=torch.tensor([b*256+i for b,L in enumerate(lengths) for i in range(L)],device='cuda',dtype=torch.int64);pages=B;kc=torch.zeros(pages,8,256,128,device='cuda',dtype=dtype);vc=torch.zeros_like(kc);out=torch.empty_like(q);table=torch.arange(B,device='cuda',dtype=torch.int32).reshape(B,1);lens=torch.tensor(lengths,device='cuda',dtype=torch.int32);append,ca=makeop('PagedCaching',[kc,vc,k,v,slots]);attn,aa=makeop('PagedAttention',[out,q,kc,vc,table,lens]);append();attn();torch.cuda.synchronize();
expected=[]
for b,L in enumerate(lengths):
 kk=kc[b,:,:L,:].float().repeat_interleave(4,0).permute(1,0,2); vv=vc[b,:,:L,:].float().repeat_interleave(4,0).permute(1,0,2); scores=torch.einsum('hd,lhd->hl',q[b].float(),kk)*(128**-.5); expected.append(torch.einsum('hl,lhd->hd',scores.softmax(-1),vv))
integration_ok=bool(torch.allclose(out.float(),torch.stack(expected),atol=.001 if a.dtype=='fp16' else .005,rtol=.01 if a.dtype=='fp16' else .05))
def both(): append();attn()
for _ in range(10): both()
torch.cuda.synchronize();rows.append({'operation':'append_attention','dtype':a.dtype,'threads':a.threads,'tokens':ntok,'correctness':'PASS' if integration_ok else 'FAIL','eager':event_measure(both),'calls_per_sample':a.calls});ca();aa()
json.dump({'threads':a.threads,'dtype':a.dtype,'env':os.environ.get('INFINIOP_PAGED_CACHING_THREADS','default'),'rows':rows},open(outdir/'results.json','w'),indent=2);json.dump({'run_id':outdir.name,'threads':a.threads,'dtype':a.dtype,'calls_per_sample':a.calls,'repeats':a.repeats,'result':'PASS' if all(r['correctness'] in ('PASS','NOT_CHECKED') for r in rows) else 'FAIL'},open(outdir/'manifest.json','w'),indent=2);check_error(lib.infiniopDestroyHandle(handle))
