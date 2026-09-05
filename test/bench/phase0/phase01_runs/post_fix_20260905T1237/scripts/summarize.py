"""Read-only summarization with explicit inputs; never rewrite raw/manifest."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--raw',type=Path,help='CSV path, or results.json; default run-dir/results.json')
    p.add_argument('--dispatch-log',type=Path)
    p.add_argument('--output',type=Path,required=True,help='NEW summary directory')
    a=p.parse_args()
    source=a.raw or a.run_dir/'results.json'
    required=[source,a.run_dir/'manifest.json']
    if a.dispatch_log:required.append(a.dispatch_log)
    missing=[str(path) for path in required if not path.is_file()]
    if missing:p.error('INCOMPLETE: missing inputs: '+', '.join(missing))
    if a.output.exists():p.error('Refusing to overwrite output: '+str(a.output))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    data=json.loads(source.read_text()) if source.suffix=='.json' else list(csv.DictReader(source.open()))
    a.output.mkdir(parents=True)
    counts=collections.Counter();latencies=collections.defaultdict(list);excluded=collections.Counter()
    for row in data:
        mode=row.get('mode',row.get('actual_execution_mode','UNKNOWN'))
        result=row.get('result',row.get('correctness','UNVERIFIED'))
        if mode=='graph_batch' and row.get('graph_correctness')!='PASS':
            result='UNVERIFIED_LEGACY_GRAPH'
        key=(row.get('dtype','UNKNOWN'),row.get('variant',row.get('requested_variant','UNKNOWN')),mode,result)
        counts[key]+=1
        if row.get('latency_us'):
            if result=='PASS' or result=='PASS_bitwise_all_cache':latencies[key[:-1]].append(float(row['latency_us']))
            else:excluded[result]+=1
    summary={'status':'COMPLETE','run_dir':str(a.run_dir.resolve()),'input_sha256':hashes,
             'correctness_counts':[dict(zip(['dtype','variant','mode','result'],key),count=count) for key,count in counts.items()],
             'excluded_latency_rows':dict(excluded),
             'timing_warning':'Only matching independently validated execution modes are eligible. No speedups or cross-case ranking inferred.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (a.output/'latency_summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dtype','variant','mode','median_us','min_us','max_us','samples'])
        # Legacy mixed-shape CSV is deliberately not pooled into latency stats.
        if len({row.get('case') for row in data})<=1:
            for key,values in latencies.items():w.writerow([*key,statistics.median(values),min(values),max(values),len(values)])
    if a.dispatch_log:
        observed=[];context='UNKNOWN'
        for line in a.dispatch_log.read_text().splitlines():
            if line.startswith('TRACE '):context=line
            if 'dispatch: path=' in line:
                observed.append({'context':context,'dispatch':dict(re.findall(r'(\w+)=([^ ]+)',line.split('dispatch: ')[1]))})
        (a.output/'observed_dispatch.csv').write_text('context,dispatch\n')
        with (a.output/'observed_dispatch.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['context','dispatch'])
            w.writerows((r['context'],json.dumps(r['dispatch'])) for r in observed)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    print('Read-only summary complete:',a.output)


if __name__=='__main__':main()
