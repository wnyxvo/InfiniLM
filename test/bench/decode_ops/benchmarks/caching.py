"""Milestone 1A.1 measured paged-caching and one-token decode workloads."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import torch
root=next(x for x in Path(__file__).resolve().parents if (x/'InfiniLM').is_dir() and (x/'InfiniCore').is_dir())
sys.path.insert(0,str(root/'InfiniCore/test/infiniop'));sys.path.insert(0,str(root/'InfiniLM/test/bench/decode_ops'))
from common.correctness_support import Native, Run, PAGE, check_output, reference
from common.timing import measure_cuda, cpu_count_test

def digest(t):
 x=t.detach().contiguous()
 if x.dtype==torch.bfloat16: x=x.view(torch.uint16)
 return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
def tensor_set_hash(ts): return hashlib.sha256(''.join(digest(x) for x in ts).encode()).hexdigest()
def parse():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',choices=['default','32','64','128','256'],default='default');p.add_argument('--grouped-heads',choices=['1','2','4'],default='1');p.add_argument('--vector',action='store_true',help='request 8-byte vector writes');p.add_argument('--dtype',choices=['fp16','bf16'],default='fp16');p.add_argument('--samples',type=int,default=20);p.add_argument('--workload-calls',type=int,default=100);p.add_argument('--batch',type=int,default=4);p.add_argument('--graph-workloads',type=int,default=20);p.add_argument('--graph-replays',type=int,default=2);p.add_argument('--round',type=int,default=1);p.add_argument('--offset-slots','--rotate-slots',dest='offset_slots',action='store_true',help='use a fixed +128 slot offset; this does not rotate slots at runtime');return p.parse_args()
def main():
 a=parse(); torch.cuda.set_device(0); run=Run(a.output); torch.manual_seed(20260905); dtype=torch.float16 if a.dtype=='fp16' else torch.bfloat16
 if a.threads=='default': os.environ.pop('INFINIOP_PAGED_CACHING_THREADS',None)
 else: os.environ['INFINIOP_PAGED_CACHING_THREADS']=a.threads
 grouped=int(a.grouped_heads)
 if grouped == 2 and a.threads != '64': raise ValueError('--grouped-heads 2 requires --threads 64')
 if grouped == 4 and a.threads != '128': raise ValueError('--grouped-heads 4 requires --threads 128')
 if grouped == 1 and a.threads == '64': raise ValueError('--threads 64 requires --grouped-heads 2')
 if a.vector:
  os.environ['INFINIOP_PAGED_CACHING_VECTOR']='1'
 else: os.environ.pop('INFINIOP_PAGED_CACHING_VECTOR',None)
 if grouped > 1: os.environ['INFINIOP_PAGED_CACHING_HEADS_PER_CTA']=str(grouped)
 else: os.environ.pop('INFINIOP_PAGED_CACHING_HEADS_PER_CTA',None)
 native=Native(); rows=[]
 for ntok in [1,32,256]:
  pages=max(1,(ntok+255)//256) + (1 if a.offset_slots else 0); k=torch.randn(ntok,8,128,device='cuda',dtype=dtype);v=torch.randn_like(k); base=torch.randn(pages,8,256,128,device='cuda',dtype=dtype); kc=base.clone();vc=torch.randn_like(base); vc0=vc.clone(); slots=(torch.arange(ntok,device='cuda',dtype=torch.int64) + (128 if a.offset_slots else 0)) % (pages*256); expected_k=base.clone();expected_v=vc0.clone()
  for i,slot in enumerate(slots.tolist()): expected_k[slot//256,:,slot%256,:]=k[i];expected_v[slot//256,:,slot%256,:]=v[i]
  call=native.operation('PagedCaching',[kc,vc,k,v,slots]); call();torch.cuda.synchronize(); ok=bool(torch.equal(kc,expected_k) and torch.equal(vc,expected_v)); in_hash=tensor_set_hash([k,v,slots,base,vc0]); ref_hash=tensor_set_hash([expected_k,expected_v]);
  def reset(): kc.copy_(base);vc.copy_(vc0)
  reset(); eager=measure_cuda(call,samples=a.samples,workload_calls=a.workload_calls,warmup=2)
  reset(); graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for _ in range(a.graph_workloads): call()
  torch.cuda.synchronize(); reset(); graphm=measure_cuda(graph.replay,samples=a.samples,workload_calls=a.graph_replays,warmup=1,formula='elapsed_ms/(graph_workloads*graph_replays)');
  for sample in graphm['samples']:
   sample['normalized_us'] = sample['elapsed_ms']*1000/(a.graph_workloads*a.graph_replays)
  vals=[x['normalized_us'] for x in graphm['samples']]; graphm.update(median_us=__import__('statistics').median(vals),min_us=min(vals),max_us=max(vals),workloads_per_graph=a.graph_workloads,graph_replays=a.graph_replays,total_logical_workloads=a.graph_workloads*a.graph_replays); reset(); graph.replay();torch.cuda.synchronize(); graph_ok=bool(torch.equal(kc,expected_k) and torch.equal(vc,expected_v)); del graph
  rows.append({'operation':'paged_caching','dtype':str(dtype),'threads':a.threads,'tokens':ntok,'target_correctness':'PASS' if ok else 'FAIL','cache_correctness':'PASS' if ok else 'FAIL','input_hash':in_hash,'expected_cache_hash':ref_hash,'eager':eager,'graph':graphm,'graph_correctness':'PASS' if graph_ok else 'FAIL','workload':'one paged_caching','workload_calls':a.workload_calls,'graph_workloads':a.graph_workloads,'graph_replays':a.graph_replays,'total_logical_workloads':a.graph_workloads*a.graph_replays,'graph_workload_calls':a.graph_workloads,'offset_slots':a.offset_slots})
 # realistic decode: history is pre-existing; each sequence appends one token once
 B=a.batch; history=256; lengths=[history]*B; q=torch.randn(B,32,128,device='cuda',dtype=dtype); new_k=torch.randn(B,8,128,device='cuda',dtype=dtype); new_v=torch.randn_like(new_k); pages=B*2; kc=torch.randn(pages,8,256,128,device='cuda',dtype=dtype);vc=torch.randn_like(kc); kc0=kc.clone();vc0=vc.clone(); slots=torch.tensor([b*512+history for b in range(B)],device='cuda',dtype=torch.int64); table=torch.tensor([[b*2,b*2+1] for b in range(B)],device='cuda',dtype=torch.int32); lens=torch.tensor([history+1]*B,device='cuda',dtype=torch.int32); out=torch.empty_like(q); ek=kc0.clone();ev=vc0.clone()
 for b,slot in enumerate(slots.tolist()): ek[slot//256,:,slot%256,:]=new_k[b];ev[slot//256,:,slot%256,:]=new_v[b]
 append=native.operation('PagedCaching',[kc,vc,new_k,new_v,slots]);attn=native.operation('PagedAttention',[out,q, kc,vc,table,lens]);append();attn();torch.cuda.synchronize(); expected=reference(q,ek,ev,table,[history+1]*B); integ=check_output(out,expected); base_hash=tensor_set_hash([q,new_k,new_v,kc0,vc0,table,lens]);
 def both(): append();attn()
 def reset_decode(): kc.copy_(kc0);vc.copy_(vc0)
 reset_decode(); eager=measure_cuda(both,samples=a.samples,workload_calls=1,warmup=2); reset_decode(); graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  for _ in range(a.graph_workloads): both()
 torch.cuda.synchronize(); reset_decode(); graphm=measure_cuda(graph.replay,samples=a.samples,workload_calls=a.graph_replays,warmup=1,formula='elapsed_ms/(graph_workloads*graph_replays)');
 for sample in graphm['samples']:
  sample['normalized_us'] = sample['elapsed_ms']*1000/(a.graph_workloads*a.graph_replays)
 vals=[x['normalized_us'] for x in graphm['samples']]; graphm.update(median_us=__import__('statistics').median(vals),min_us=min(vals),max_us=max(vals),workloads_per_graph=a.graph_workloads,graph_replays=a.graph_replays,total_logical_workloads=a.graph_workloads*a.graph_replays); reset_decode(); graph.replay();torch.cuda.synchronize(); expected2=reference(q,ek,ev,table,[history+1]*B); graph_ok=check_output(out,expected2)['attention_correctness']=='PASS'; del graph
 rows.append({'operation':'append_attention','dtype':str(dtype),'threads':a.threads,'batch':B,'history_tokens':history,'append_tokens':B,'target_correctness':integ['attention_correctness'],'cache_correctness':'PASS' if torch.equal(kc,ek) and torch.equal(vc,ev) else 'FAIL','graph_correctness':'PASS' if graph_ok else 'FAIL','input_hash':base_hash,'workload':'one append + one attention','workload_calls':1,'graph_workloads':a.graph_workloads,'graph_replays':a.graph_replays,'total_logical_workloads':a.graph_workloads*a.graph_replays,'eager':eager,'graph':graphm})
 actual_threads = 1024 if a.threads == 'default' else int(a.threads)
 run.add(rows=rows,cpu_count_test=cpu_count_test(),requested_threads=a.threads,actual_threads=actual_threads,vector_requested=a.vector,vector_active=bool(a.vector and a.threads == '32' and grouped == 1),heads_per_cta=grouped,graph_workloads=a.graph_workloads,graph_replays=a.graph_replays,total_logical_workloads=a.graph_workloads*a.graph_replays,dtype=str(dtype),script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),result='PASS' if all(r['target_correctness']=='PASS' and r['cache_correctness']=='PASS' and r['graph_correctness']=='PASS' for r in rows) else 'FAIL')
 native.close();return run.finish()
if __name__=='__main__':sys.exit(main())
