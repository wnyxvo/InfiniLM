"""Correctness matrix for the Milestone 2B shared-KV GQA Split-KV path."""
import argparse, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, PAGE, reference, check_output, configure

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    torch.cuda.set_device(0); run = Run(args.output)
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for num_splits in (2, 4, 8):
                for lengths in ([1],[7],[255],[256],[257],[2049],[7,256,2049]):
                    batch = len(lengths); cap = (max(lengths)+PAGE-1)//PAGE
                    torch.manual_seed(20260906 + batch + max(lengths))
                    q = torch.randn(batch,32,128,device='cuda',dtype=dtype)
                    k = torch.randn(batch*cap,8,PAGE,128,device='cuda',dtype=dtype); v = torch.randn_like(k)
                    table = torch.randperm(batch*cap,device='cuda',dtype=torch.int32).reshape(batch,cap)
                    lens = torch.tensor(lengths,device='cuda',dtype=torch.int32); out = torch.full_like(q,float('nan'))
                    n = Native(); call = n.operation('PagedAttention',[out,q,k,v,table,lens])
                    configure('default')
                    os.environ.update({'INFINIOP_FLASH_DECODE_KERNEL':'cta','INFINIOP_FLASH_DECODE_SPLITKV':'1',
                        'INFINIOP_FLASH_NUM_SPLITS':str(num_splits),'INFINIOP_FLASH_GQA_SHARED_SPLITKV':'1',
                        'INFINIOP_FLASH_GQA_SPLITKV':'0','INFINIOP_FLASH_GQA_FUSED':'0',
                        'INFINIOP_FLASH_DEBUG_DISPATCH':'1'})
                    ref1 = reference(q,k,v,table,lengths)
                    out.fill_(float('nan')); call(); torch.cuda.synchronize(); eager = check_output(out,ref1)
                    # Capture one workload. Inputs, cache values and lengths are updated in-place for replay.
                    graph = torch.cuda.CUDAGraph(); base_q=q.clone(); base_k=k.clone(); base_v=v.clone()
                    with torch.cuda.graph(graph): call()
                    # Negative control: without replay, output remains sentinel.
                    out.fill_(float('nan')); torch.cuda.synchronize(); skip_ok = bool(torch.isnan(out).all().item())
                    q2 = torch.randn_like(q); k2 = torch.randn_like(k); v2 = torch.randn_like(v)
                    q.copy_(q2); k.copy_(k2); v.copy_(v2); lens2 = torch.tensor([max(1, x-1) for x in lengths],device='cuda',dtype=torch.int32); lens.copy_(lens2)
                    graph.replay(); torch.cuda.synchronize(); ref2 = reference(q2,k2,v2,table,lens2.tolist()); graph_ok = check_output(out,ref2)
                    run.add(dtype=str(dtype), lengths=lengths, batch=batch, capacity_pages=cap, num_splits=num_splits,
                        dispatch='splitkv_gqa_shared', eager_correctness=eager['attention_correctness'],
                        graph_correctness=graph_ok['attention_correctness'], graph_update_correctness=graph_ok['attention_correctness'],
                        skip_replay_negative_control='PASS' if skip_ok else 'FAIL', eager_metrics=eager, graph_metrics=graph_ok,
                        result='PASS' if eager['attention_correctness']=='PASS' and graph_ok['attention_correctness']=='PASS' and skip_ok else 'FAIL')
                    del graph; n.close()
        return run.finish()
    except Exception as e:
        run.finish(e); raise
if __name__ == '__main__': raise SystemExit(main())
