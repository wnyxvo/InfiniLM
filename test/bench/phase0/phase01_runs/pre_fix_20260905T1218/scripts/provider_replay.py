"""C++ native/FlashAttention provider and actual InfiniCore Graph smoke test.

Run under nsys for capture/kernel evidence. No latency claims from this script.
"""
import argparse
import json
import os
from pathlib import Path
import torch
import infinicore as ic
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.mha_kvcache import mha_kvcache

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--flash',action='store_true')
a=p.parse_args()
torch.manual_seed(20260905)
ic.set_device(ic.device('cuda',0))
os.environ['INFINICORE_GRAPH_DEBUG']='1'
os.environ['INFINIOP_FLASH_DEBUG_DISPATCH']='1'
for key in ['DECODE_SPLITKV','NUM_SPLITS','GQA_FUSED','DECODE_KERNEL']:
    os.environ.pop('INFINIOP_FLASH_'+key,None)
q=torch.randn(2,32,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(8,8,256,128,device='cuda',dtype=q.dtype)
v=torch.randn_like(k)
newk=torch.randn(2,8,128,device='cuda',dtype=q.dtype)
newv=torch.randn_like(newk)
bt=torch.tensor([[3,0,2,1],[7,4,6,5]],device='cuda',dtype=torch.int32)
lens=torch.tensor([255,257],device='cuda',dtype=torch.int32)
slots=torch.tensor([3*256+254,4*256],device='cuda',dtype=torch.int64)
out=torch.empty_like(q)
if a.flash:
    k=k.permute(0,2,1,3).contiguous();v=v.permute(0,2,1,3).contiguous()

def wrap(t):
    # from_torch currently assumes contiguous; use explicit strides for views.
    return ic.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),dtype={torch.bfloat16:ic.bfloat16,torch.int32:ic.int32,torch.int64:ic.int64}[t.dtype],device=ic.device('cuda',0))

kc=k.permute(0,2,1,3) if a.flash else k
vc=v.permute(0,2,1,3) if a.flash else v
tensors=[wrap(t) for t in [q,kc,vc,newk,newv,bt,lens,slots,out]]
iq,ik,iv,ink,inv,ibt,ilens,islots,io=tensors
fq,fk,fv,fo=[wrap(t) for t in [q.unsqueeze(1),k,v,out.unsqueeze(1)]]

def ops():
    paged_caching(ik,iv,ink,inv,islots)
    if a.flash:mha_kvcache(fq,fk,fv,ilens,ibt,scale=128**-.5,out=fo)
    else:paged_attention(iq,ik,iv,ibt,ilens,scale=128**-.5,out=io)

def verify():
    torch.cuda.synchronize()
    expected=[]
    for b,L in enumerate(lens.tolist()):
        ids=bt[b,:((L+255)//256)].long()
        kk=kc[ids].permute(0,2,1,3).reshape(-1,8,128)[:L].float().repeat_interleave(4,1)
        vv=vc[ids].permute(0,2,1,3).reshape(-1,8,128)[:L].float().repeat_interleave(4,1)
        score=torch.einsum('hd,lhd->hl',q[b].float(),kk)*128**-.5
        expected.append(torch.einsum('hl,lhd->hd',score.softmax(-1),vv))
    ref=torch.stack(expected)
    assert torch.allclose(out.float(),ref,atol=.005,rtol=.05)
    return (out.float()-ref).abs().max().item()

torch.cuda.synchronize();ops();ic.sync_stream();errors=[verify()]
ic.start_graph_recording();ops();graph=ic.stop_graph_recording()
for step in range(3):
    lengths=[256+step,258+step]
    lens.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
    if step==1:bt.copy_(bt.roll(1,1))
    s=[int(bt[b,(L-1)//256])*256+(L-1)%256 for b,L in enumerate(lengths)]
    slots.copy_(torch.tensor(s,device='cuda',dtype=torch.int64))
    newk.normal_();newv.normal_();torch.cuda.synchronize()
    graph.run();ic.sync_stream();errors.append(verify())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'requested_provider':'flash' if a.flash else 'native','requested_graph_mode':'InfiniCore Graph','correctness':'PASS','replay_steps':3,'max_abs_errors':errors,'actual_execution_mode':'see Graph debug and Nsight evidence','loaded_libraries':sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if '/libinfini' in l or '/libflash-attn' in l})},indent=2))
print(a.output.read_text())
