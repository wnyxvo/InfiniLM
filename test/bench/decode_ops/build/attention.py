"""Build the hd128 paged-attention object with the experimental GQA split dispatch."""
import argparse, hashlib, json, re, shutil, subprocess, sys
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recipe(path):
 t=Path(path).read_text(); vals=t.split('values = {',1)[1].split('depfiles =',1)[0]
 q=lambda x:[json.loads('"'+v+'"') for v in re.findall(r'"((?:\\.|[^"\\])*)"',x)]
 files=t.split('files = {',1)[1].split('}',1)[0] if 'files = {' in t else ''
 return q(files),q(vals)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--core',type=Path,required=True); p.add_argument('--prefix',type=Path,required=True); p.add_argument('--base-prefix',type=Path,required=True); p.add_argument('--cache-prefix',type=Path); p.add_argument('--evidence',type=Path,required=True); a=p.parse_args()
 core=a.core.resolve(); base=a.base_prefix.resolve(); cache=a.cache_prefix.resolve() if a.cache_prefix else None; a.prefix.mkdir(parents=True,exist_ok=False); a.evidence.mkdir(parents=True,exist_ok=False)
 libdir=a.prefix/'lib'; objdir=a.prefix/'objects'; libdir.mkdir(); objdir.mkdir()
 for lib in Path('/root/.infini/lib').glob('*.so'):
  if lib.name!='libinfiniop.so': (libdir/lib.name).symlink_to(lib)
 stem='infiniop-nvidia/linux/x86_64/release'; cu='src/infiniop/ops/paged_attention/nvidia/paged_attention_hd128.cu'; oldobj='build/.objs/'+stem+'/'+cu+'.o'; newobj=objdir/'paged_attention_hd128.cu.o'; dlink='rules/cuda/devlink/infiniop-nvidia_gpucode.cu.o'; newlink=objdir/'infiniop-nvidia_gpucode.cu.o'; dep=core/'build/.deps'
 m={'command':[sys.executable,*sys.argv],'core':str(core),'core_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=core,text=True).strip(),'source_hashes':{cu:sha(core/cu),'src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh':sha(core/'src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh')},'prefix':str(a.prefix),'base_prefix':str(base),'cache_prefix':str(cache) if cache else None,'commands':[]}
 (a.evidence/'InfiniCore.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=core)); log=(a.evidence/'build.log').open('w')
 def run(cmd):
  r=subprocess.run(cmd,cwd=core,stdout=log,stderr=subprocess.STDOUT); m['commands'].append({'argv':cmd,'exit_code':r.returncode}); (a.evidence/'build.json').write_text(json.dumps(m,indent=2)); r.check_returncode()
 _,flags=recipe(dep/stem/(cu+'.o.d')); arch=[x for x in flags if x.startswith('-gencode=')]; run([flags[0],'-c',*flags[1:],cu,'-o',str(newobj)])
 inputs,lf=recipe(dep/(stem+'/'+dlink+'.d'))
 attention_names=('paged_attention_hd64.cu.o','paged_attention_hd128.cu.o','paged_attention_hd192.cu.o','paged_attention_hd256.cu.o','paged_attention_hd576.cu.o','paged_attention_mla_hd576_v512.cu.o')
 def remap(x):
  if cache and x.endswith('/paged_caching_nvidia.cu.o'): return str(cache/'objects'/'paged_caching_nvidia.cu.o')
  return str(base/'objects'/Path(x).name) if any(x.endswith('/'+n) for n in attention_names) else x
 inputs=[str(newobj) if x==oldobj else remap(x) for x in inputs]; run([lf[0],*inputs,*lf[1:],*arch,'-o',str(newlink)])
 archive=libdir/'libinfiniop-nvidia.a'; shutil.copy2(base/'lib/libinfiniop-nvidia.a',archive); run(['/usr/bin/ar','r',str(archive),str(newobj),str(newlink)])
 inputs,lf=recipe(dep/'infiniop/linux/x86_64/release/libinfiniop.so.d'); objs=[str(newobj) if x==oldobj else remap(x) for x in inputs if x.endswith('.o')]; run([lf[0],*objs,'-L'+str(libdir),*lf[1:],'-o',str(libdir/'libinfiniop.so')]); m['new_library_hash']=sha(libdir/'libinfiniop.so'); m['exit_code']=0; (a.evidence/'build.json').write_text(json.dumps(m,indent=2)); print(m['new_library_hash'])
if __name__=='__main__': main()
