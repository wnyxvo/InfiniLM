"""Qwen3-4B paged service request matrix for Milestone 5.
The server is started separately so A/B workers never share a GPU.
"""
import argparse, concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
PROMPTS = {
    'smoke_zh': '请用一句话介绍 CUDA kernel 优化。 /no_think',
    'english': 'Explain one practical CUDA kernel optimization in one sentence. /no_think',
    'long_zh': ('请分析大模型推理中 KV cache、分页注意力和内存带宽之间的关系，给出清晰的工程建议。 ' * 90) + '/no_think',
}
def one(port, name, prompt, max_tokens):
    body={'model':'master','messages':[{'role':'user','content':prompt}],
          'max_tokens':max_tokens,'temperature':1.0,'top_p':0.8,'top_k':1,'stream':False}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    t=time.perf_counter(); status=None; raw=''; err=None
    try:
        with urllib.request.urlopen(req,timeout=180) as r: status=r.status; raw=r.read().decode()
        obj=json.loads(raw); choice=obj.get('choices',[{}])[0]
        usage=obj.get('usage',{}); content=choice.get('message',{}).get('content','')
        return {'name':name,'http_status':status,'response':obj,'finish_reason':choice.get('finish_reason'),
                'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),
                'total_tokens':usage.get('total_tokens'),'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
                'latency_ms':(time.perf_counter()-t)*1000,'error':None}
    except Exception as e:
        err=repr(e)
        return {'name':name,'http_status':status,'raw':raw,'latency_ms':(time.perf_counter()-t)*1000,'error':err}
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=3);p.add_argument('--max-tokens',type=int,default=16);a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False); rows=[]
    # Warmup is intentionally separate from measured rounds.
    one(a.port,'warmup',PROMPTS['smoke_zh'],a.max_tokens)
    for rnd in range(1,a.rounds+1):
        for name,prompt in PROMPTS.items(): rows.append({'round':rnd,'concurrency':1,**one(a.port,name,prompt,a.max_tokens)})
        for n in (4,16):
            prompt_items=list(PROMPTS.items())
            prompts=[(f'{prompt_items[i % len(prompt_items)][0]}_{i}', prompt_items[i % len(prompt_items)][1]) for i in range(n)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                fut=[ex.submit(one,a.port,name,prompt,a.max_tokens) for name,prompt in prompts]
                for f in fut: rows.append({'round':rnd,'concurrency':n,**f.result()})
    result='PASS' if all(r.get('http_status')==200 and not r.get('error') and r.get('finish_reason') in ('stop','length') for r in rows) else 'FAIL'
    (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps({'port':a.port,'rounds':a.rounds,'max_tokens':a.max_tokens,'prompt_names':list(PROMPTS),'result':result,'rows':len(rows)},indent=2))
    print(json.dumps({'result':result,'rows':len(rows)})); return 0 if result=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
