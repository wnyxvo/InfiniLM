"""Shared CUDA-event timing and workload-count helpers."""
import statistics
import torch


def run_workload(workload, calls):
    """Execute exactly ``calls`` logical workloads and return the count."""
    for _ in range(calls):
        workload()
    return calls


def measure_cuda(workload, *, samples, workload_calls, warmup=0, formula='elapsed_ms/workload_calls'):
    run_workload(workload, warmup)
    torch.cuda.synchronize()
    rows = []
    for i in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run_workload(workload, workload_calls)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        rows.append({'sample': i, 'elapsed_ms': elapsed_ms, 'workload_calls': workload_calls,
                     'formula': formula, 'normalized_us': elapsed_ms * 1000 / workload_calls})
    vals = [r['normalized_us'] for r in rows]
    return {'samples': rows, 'median_us': statistics.median(vals), 'min_us': min(vals),
            'max_us': max(vals), 'workload_calls': workload_calls, 'formula': formula}


def cpu_count_test():
    observed = {}
    for n in (1, 100):
        count = {'value': 0}
        def work():
            count['value'] += 1
        observed[n] = run_workload(work, n)
        assert count['value'] == n
    return {'calls_1': observed[1], 'calls_100': observed[100], 'status': 'PASS'}
