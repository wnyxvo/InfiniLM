"""Recompile only hd128 and relink from existing XMake objects into a NEW prefix.

No install, dependency download, global override, or mutation of old build objects.
The existing XMake .d files are used as recorded compiler/linker flags.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def recipe(path):
    text=path.read_text()
    files=re.search(r'files\s*=\s*\{(.*?)\}',text,re.S)
    values=text.split('values = {',1)[1].split('depfiles =',1)[0]
    quoted=lambda s:[json.loads('"'+x+'"') for x in re.findall(r'"((?:\\.|[^"\\])*)"',s)]
    return quoted(files.group(1)) if files else [],quoted(values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--core',type=Path,default=Path(__file__).resolve().parents[4]/'InfiniCore')
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args(); core=a.core.resolve(); prefix=a.prefix.resolve()
    prefix.mkdir(parents=True,exist_ok=False)
    a.evidence.mkdir(parents=True,exist_ok=False)
    libdir=prefix/'lib';libdir.mkdir()
    objdir=prefix/'objects';objdir.mkdir()
    for lib in Path('/root/.infini/lib').glob('*.so'):
        if lib.name!='libinfiniop.so':(libdir/lib.name).symlink_to(lib)
    dep=core/'build/.deps'
    stem='infiniop-nvidia/linux/x86_64/release'
    cus=[
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_hd64.cu',
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_hd128.cu',
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_hd192.cu',
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_hd256.cu',
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_hd576.cu',
        'src/infiniop/ops/paged_attention/nvidia/paged_attention_mla_hd576_v512.cu',
    ]
    dlink='rules/cuda/devlink/infiniop-nvidia_gpucode.cu.o'
    oldobjs=['build/.objs/'+stem+'/'+cu+'.o' for cu in cus]
    newobjs=[objdir/(Path(cu).name+'.o') for cu in cus]
    newlink=objdir/'infiniop-nvidia_gpucode.cu.o'
    manifest={'command':[sys.executable,*sys.argv],'core':str(core),'prefix':str(prefix),
              'core_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=core,text=True).strip(),
              'source_hash':sha(core/'src/infiniop/ops/paged_attention/cuda/kernel_v2.cuh'),
              'old_library_hash':sha(core/'build/linux/x86_64/release/libinfiniop.so'),
              'commands':[],'inputs':{}}
    (a.evidence/'InfiniCore.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=core))
    log=(a.evidence/'build.log').open('w')
    def run(cmd):
        print('RUN',cmd[0],flush=True)
        proc=subprocess.run(cmd,cwd=core,stdout=log,stderr=subprocess.STDOUT)
        manifest['commands'].append({'argv':cmd,'exit_code':proc.returncode})
        (a.evidence/'build.json').write_text(json.dumps(manifest,indent=2))
        proc.check_returncode()
    arch_flags=[]
    for cu,newobj in zip(cus,newobjs):
        _,flags=recipe(dep/stem/(cu+'.o.d'))
        if not arch_flags: arch_flags=[flag for flag in flags if flag.startswith('-gencode=')]
        if not arch_flags: raise RuntimeError('Missing compile architecture')
        run([flags[0],'-c',*flags[1:],cu,'-o',str(newobj)])
    inputs,flags=recipe(dep/stem/(dlink+'.d'))
    for name in inputs: manifest['inputs'][name]=sha(core/name)
    for oldobj,newobj in zip(oldobjs,newobjs):
        inputs=[str(newobj) if name==oldobj else name for name in inputs]
    run([flags[0],*inputs,*flags[1:],*arch_flags,'-o',str(newlink)])
    archive=libdir/'libinfiniop-nvidia.a'
    shutil.copy2(core/'build/linux/x86_64/release/libinfiniop-nvidia.a',archive)
    run(['/usr/bin/ar','r',str(archive),*(str(x) for x in newobjs),str(newlink)])
    inputs,flags=recipe(dep/'infiniop/linux/x86_64/release/libinfiniop.so.d')
    for name in inputs:manifest['inputs'][name]=sha(core/name)
    objects=[name for name in inputs if name.endswith('.o')]
    run([flags[0],*objects,'-L'+str(libdir),*flags[1:],'-o',str(libdir/'libinfiniop.so')])
    manifest['new_library_hash']=sha(libdir/'libinfiniop.so')
    manifest['exit_code']=0
    (a.evidence/'build.json').write_text(json.dumps(manifest,indent=2))
    print(manifest['new_library_hash'])


if __name__=='__main__':main()
