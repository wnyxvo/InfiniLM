"""Read-only summarization with explicit inputs; never rewrite raw/manifest."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def derive(row):
    """Return explicit target/control/cache/overall fields without mutating raw data."""
    mode=row.get('mode', row.get('actual_execution_mode', 'UNKNOWN'))
    negative=bool(row.get('negative_control'))
    target=row.get('target_correctness')
    if target is None:
        target=row.get('graph_correctness') if mode in ('torch_cuda_graph','infinicore_graph','graph') else row.get('attention_correctness')
    if target is None: target=row.get('correctness')
    control=row.get('control_correctness', row.get('crosscheck_split4'))
    cache=row.get('cache_correctness')
    detected=str(row.get('negative_control_detected', row.get('detected', ''))).lower() == 'true'
    recorded=row.get('result', row.get('overall_result', row.get('correctness', 'UNVERIFIED')))
    if mode == 'graph_batch' and row.get('graph_correctness') != 'PASS':
        overall='UNVERIFIED_LEGACY_GRAPH'
    elif negative:
        overall='PASS' if detected else ('FAIL' if detected is False else 'INCOMPLETE')
    else:
        checks=[target, cache, control]
        if any(v in ('FAIL','ERROR') for v in checks): overall='FAIL'
        elif any(v in ('NOT_RUN','UNVERIFIED','INCOMPLETE',None) for v in checks if v is not None or target is None):
            overall='INCOMPLETE'
        else: overall='PASS'
    inconsistent=(recorded not in ('',None) and recorded != overall)
    return {'test_kind':'negative_control' if negative else 'positive', 'target_correctness':target or 'NOT_RUN',
            'control_correctness':control or 'NOT_RUN', 'cache_correctness':cache or 'NOT_RUN',
            'overall_result':overall, 'negative_control_detected':detected if negative else None,
            'recorded_result':recorded, 'inconsistent':inconsistent, 'mode':mode}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--raw',type=Path,help='CSV path, or results.json; default run-dir/results.json')
    p.add_argument('--dispatch-log',type=Path)
    p.add_argument('--output',type=Path,required=True,help='NEW summary directory')
    a=p.parse_args()
    source=a.raw or a.run_dir/'results.json'
    required=[source,a.run_dir/'manifest.json']
    if a.dispatch_log: required.append(a.dispatch_log)
    missing=[str(path) for path in required if not path.is_file()]
    if missing: p.error('INCOMPLETE: missing inputs: '+', '.join(missing))
    if a.output.exists(): p.error('Refusing to overwrite output: '+str(a.output))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    data=json.loads(source.read_text()) if source.suffix=='.json' else list(csv.DictReader(source.open()))
    a.output.mkdir(parents=True)
    counts=collections.Counter(); latencies=collections.defaultdict(list); excluded=collections.Counter(); inconsistencies=[]
    for idx,row in enumerate(data):
        d=derive(row)
        variant=row.get('variant',row.get('requested_variant','UNKNOWN')); dtype=row.get('dtype','UNKNOWN')
        key=(d['test_kind'],dtype,variant,d['mode'],d['overall_result'])
        counts[key]+=1
        if d['inconsistent']:
            inconsistencies.append({'index':idx,'recorded_result':d['recorded_result'],**{k:d[k] for k in ('target_correctness','control_correctness','cache_correctness','overall_result','mode')}})
        if row.get('latency_us'):
            if d['overall_result'] in ('PASS','PASS_bitwise_all_cache'): latencies[key[:-1]].append(float(row['latency_us']))
            else: excluded[d['overall_result']]+=1
    summary={'status':'COMPLETE','run_dir':str(a.run_dir.resolve()),'input_sha256':hashes,
             'correctness_counts':[dict(zip(['test_kind','dtype','variant','mode','result'],key),count=count) for key,count in counts.items()],
             'excluded_latency_rows':dict(excluded), 'inconsistency_count':len(inconsistencies), 'inconsistencies':inconsistencies,
             'timing_warning':'Only independently validated overall PASS rows are eligible. No speedups or cross-case ranking inferred.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'verdicts.json').write_text(json.dumps([{'index':i,**derive(r)} for i,r in enumerate(data)],indent=2))
    with (a.output/'latency_summary.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['test_kind','dtype','variant','mode','median_us','min_us','max_us','samples'])
        if len({row.get('case') for row in data})<=1:
            for key,values in latencies.items(): w.writerow([*key,statistics.median(values),min(values),max(values),len(values)])
    if a.dispatch_log:
        observed=[]; context='UNKNOWN'
        for line in a.dispatch_log.read_text().splitlines():
            if line.startswith('TRACE '): context=line
            if 'dispatch: path=' in line: observed.append({'context':context,'dispatch':dict(re.findall(r'(\w+)=([^ ]+)',line.split('dispatch: ')[1]))})
        with (a.output/'observed_dispatch.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['context','dispatch']);w.writerows((r['context'],json.dumps(r['dispatch'])) for r in observed)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    print('Read-only summary complete:',a.output,'inconsistencies=',len(inconsistencies))


if __name__=='__main__':main()
