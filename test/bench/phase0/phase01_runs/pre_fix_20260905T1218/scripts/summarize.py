"""Attach observed dispatch paths to raw rows and preserve evidence summaries."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import statistics

root=Path(__file__).resolve().parent
logs={'native_20260905':'/tmp/phase0-native.log','gqa_repro_20260905':'/tmp/phase0-gqa.log','gqa_fp16_20260905':'/tmp/phase0-gqa-fp16.log','caching_aligned_20260905':'/tmp/phase0-caching-aligned.log','smoke_20260905':'/tmp/phase0-smoke.log'}
for run,log in logs.items():
    dest=root/'runs'/run
    shutil.copyfile(log,dest/'dispatch.log')
    paths={};case=None
    for line in Path(log).read_text().splitlines():
        if line.startswith('TRACE '):case=tuple(line.split()[1:3])
        if 'dispatch: path=' in line and case:
            paths[case]=dict(re.findall(r'(\w+)=([^ ]+)',line.split('dispatch: ')[1]))
    raw=dest/'raw.csv'
    with raw.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for r in rows:
        if r['op']=='attention':
            observed=paths.get((r['case'],r['requested_variant']))
            r['actual_kernel']=json.dumps(observed) if observed else 'UNKNOWN; dispatch signature suppressed; see log'
        else:r['actual_kernel']='pagedCaching<Tdata,1024>; BF16 confirmed by Nsight; F16/F32 SOURCE_CONFIRMED'
    with raw.open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    groups=collections.defaultdict(list)
    for r in rows:
        if r['latency_us']:groups[(r['op'],r['case'],r['dtype'],r['requested_variant'],r['actual_execution_mode'])].append(float(r['latency_us']))
    with (dest/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['run_id','op','case','dtype','variant','mode','median_us','min_us','max_us','stdev_us','samples'])
        for key,v in groups.items():w.writerow([run,*key,statistics.median(v),min(v),max(v),statistics.stdev(v),len(v)])
    m=json.loads((dest/'manifest.json').read_text())
    m['artifact_script_sha256_at_finalization']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    m['provenance_limit']='Installed libinfiniop hash equals local build artifact; no embedded source SHA/build attestation. Initial repositories clean, later dirty paths are these Phase0 artifacts.'
    (dest/'manifest.json').write_text(json.dumps(m,indent=2))

dest=root/'runs/native_20260905'
for name,src in {'upstream_caching.log':'/tmp/phase0-upstream-caching.log','upstream_attention.log':'/tmp/phase0-upstream-attention.log','provider_native.log':'/tmp/phase0-provider-native-v2.log','provider_flash.log':'/tmp/phase0-provider-flash.log','provider_nodes.log':'/tmp/phase0-provider-nodes.log'}.items():
    shutil.copyfile(src,dest/name)
db=Path('/tmp/phase0-provider-nodes.sqlite')
if db.exists():
    c=sqlite3.connect(db)
    query='''select s.value,k.graphNodeId,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.end-k.start as duration_ns from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where s.value like '%pagedCaching%' or s.value like '%flashAttentionDecodeHd128%' '''
    cur=c.execute(query)
    with (dest/'graph_nodes.csv').open('w') as f:
        w=csv.writer(f);w.writerow([x[0] for x in cur.description]);w.writerows(cur.fetchall())
    evidence={}
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        cols=[r[1] for r in c.execute('pragma table_info('+table+')')]
        evidence[table]={'columns':cols}
        if 'graphNodeId' in cols:evidence[table]['graph_node_count']=c.execute('select count(*) from '+table+' where graphNodeId is not null and graphNodeId != 0').fetchone()[0]
    (dest/'graph_evidence.json').write_text(json.dumps(evidence,indent=2))
print('summaries and observed dispatch paths saved')
