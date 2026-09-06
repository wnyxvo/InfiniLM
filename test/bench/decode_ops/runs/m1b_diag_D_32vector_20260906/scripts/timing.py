"""Shared CUDA-event timing and workload-count helpers."""
import statistics, time
import torch

def measure_cuda(workload, *, samples, workload_calls, warmup=0, formula='elapsed_ms/workload_calls'):
    for _ in range(warmup): workload()
    torch.cuda.synchronize()
    rows=[]
    for i in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); workload(); end.record(); end.synchronize()
        elapsed_ms=start.elapsed_time(end)
        rows.append({'sample':i,'elapsed_ms':elapsed_ms,'workload_calls':workload_calls,'formula':formula,'normalized_us':elapsed_ms*1000/workload_calls})
    vals=[r['normalized_us'] for r in rows]
    return {'samples':rows,'median_us':statistics.median(vals),'min_us':min(vals),'max_us':max(vals),'workload_calls':workload_calls,'formula':formula}

def cpu_count_test():
    for n in (1,100):
        count={'value':0}
        def work(): count['value']+=1
        for _ in range(n): work()
        assert count['value']==n
    return {'calls_1':1,'calls_100':100,'status':'PASS'}
