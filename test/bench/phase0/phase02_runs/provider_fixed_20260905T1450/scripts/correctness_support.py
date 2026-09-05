"""Phase 0.1 test-only native bindings, references, provenance and dispatch checks."""
import contextlib
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback

import torch

LM = Path(__file__).resolve().parents[3]
CORE = LM.parent / 'InfiniCore'
SEED = 20260905
PAGE = 256
sys.path.insert(0, str(CORE / 'test/infiniop'))
from libinfiniop import (LIBINFINIOP as lib, check_error,
    infiniopTensorDescriptor_t, infiniopOperatorDescriptor_t, infiniopHandle_t)

BASE = dict(DECODE_KERNEL='cta', GQA_FUSED='1', CTA_TILE='8', CTA_THREADS='64',
            DECODE_SPLITKV='1', NUM_SPLITS='4', DEBUG_DISPATCH='1', DEBUG_SPLITS='1')
VARIANTS = {
    'default': {}, 'gqa': {'DECODE_SPLITKV': '0'},
    'auto': {'DECODE_SPLITKV': 'auto'},
    'cta': {'DECODE_SPLITKV': '0', 'GQA_FUSED': '0'},
    'split2': {'NUM_SPLITS': '2'},
}


def configure(variant):
    config = BASE | VARIANTS[variant]
    for key in list(os.environ):
        if key.startswith('INFINIOP_FLASH_'):
            del os.environ[key]
    for key, value in config.items():
        os.environ['INFINIOP_FLASH_' + key] = value
    return config


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def command(argv, cwd=None):
    p = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    return {'command': argv, 'exit_code': p.returncode,
            'stdout': p.stdout, 'stderr': p.stderr}


class Run:
    def __init__(self, output):
        self.path = Path(output).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.manifest = {'run_id': self.path.name, 'command': [sys.executable, *sys.argv],
                         'seed': SEED, 'python': sys.version, 'torch': torch.__version__,
                         'torch_path': torch.__file__, 'cuda': torch.version.cuda,
                         'repos': {}, 'scripts': {}, 'exit_code': None,
                         'initial_environment': {k: v for k, v in os.environ.items()
                             if k.startswith(('INFINIOP_FLASH_', 'INFINICORE_GRAPH', 'INFINICORE_DISABLE_DEVICE_GRAPH'))
                             or k in ('INFINI_ROOT','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','PYTHONPATH')}}
        for name, repo in [('InfiniLM', LM), ('InfiniCore', CORE)]:
            self.manifest['repos'][name] = {
                'path': str(repo), 'sha': command(['git','rev-parse','HEAD'],repo),
                'dirty': command(['git','status','--short'],repo),
                'branch': command(['git','branch','--show-current'],repo),
                'submodules': command(['git','submodule','status'],repo)}
            (self.path / (name + '.patch')).write_text(command(['git','diff','--binary','HEAD'],repo)['stdout'])
        snapshots = self.path / 'scripts'
        snapshots.mkdir()
        for script in Path(__file__).parent.glob('*.py'):
            self.manifest['scripts'][script.name] = digest(script)
            (snapshots / script.name).write_bytes(script.read_bytes())
        self.manifest['build_config'] = (CORE / '.xmake/linux/x86_64/xmake.conf').read_text()
        self.manifest['device'] = str(torch.cuda.get_device_properties(0))
        self.manifest['nvidia_smi'] = command(['nvidia-smi'])
        self.manifest['nvcc'] = command(['nvcc','--version'])
        self.save()

    def add(self, **row):
        self.rows.append({'run_id': self.path.name, **row})
        self.save()

    def save(self):
        self.manifest['loaded_libraries'] = {
            p: digest(p) for p in sorted({line.split()[-1]
                for line in Path('/proc/self/maps').read_text().splitlines()
                if '/libinfini' in line or '/libflash-attn' in line or '/_infinicore' in line or '/_infinilm' in line}) if Path(p).is_file()}
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest, indent=2))
        (self.path / 'results.json').write_text(json.dumps(self.rows, indent=2, allow_nan=False))

    def finish(self, error=None):
        failed = error is not None or any(r.get('result') == 'FAIL' for r in self.rows)
        self.manifest['exit_code'] = int(failed)
        self.manifest['result'] = 'FAIL' if failed else 'PASS'
        if error:
            self.manifest['exception'] = str(error)
            (self.path/'exception.log').write_text(traceback.format_exc())
        self.save()
        return int(failed)


@contextlib.contextmanager
def capture_stderr():
    """Capture C fprintf as well as Python stderr; no production tracing edits."""
    with tempfile.TemporaryFile(mode='w+b') as f:
        original = os.dup(2)
        try:
            os.dup2(f.fileno(), 2)
            yield f
        finally:
            C.CDLL(None).fflush(None)
            os.dup2(original, 2)
            os.close(original)


def trace_call(call, variant):
    # Existing debug suppresses repeated signatures. Prime a different path so
    # the target path must emit a fresh signature, then restore complete knobs.
    configure('cta' if variant in ('default','split2') else 'default')
    call()
    torch.cuda.synchronize()
    config = configure(variant)
    with capture_stderr() as f:
        call()
        torch.cuda.synchronize()
        C.CDLL(None).fflush(None)
        f.seek(0)
        trace = f.read().decode()
    lines = [line for line in trace.splitlines() if 'dispatch: path=' in line]
    if not lines:
        raise RuntimeError('No runtime dispatch evidence: ' + trace)
    dispatch = dict(re.findall(r'(\w+)=([^ ]+)', lines[-1].split('dispatch: ')[1]))
    return config, dispatch, trace


def reference(q, k, v, table, lengths):
    """Independent FP32 attention; cache contains all tokens through current step."""
    output = []
    for b, length in enumerate(lengths):
        ids = table[b, :(length + PAGE - 1)//PAGE].long()
        keys = k[ids].permute(0,2,1,3).reshape(-1,k.shape[1],128)[:length].float()
        vals = v[ids].permute(0,2,1,3).reshape(-1,v.shape[1],128)[:length].float()
        keys = keys.repeat_interleave(q.shape[1]//k.shape[1],1)
        vals = vals.repeat_interleave(q.shape[1]//k.shape[1],1)
        scores = torch.einsum('hd,lhd->hl',q[b].float(),keys) * 128**-.5
        output.append(torch.einsum('hl,lhd->hd',scores.softmax(-1),vals))
    return torch.stack(output)


def check_output(out, expected):
    finite = torch.isfinite(out)
    error = (out.float()-expected).abs()
    head_errors = error.amax(dim=(0,2)).tolist()
    atol, rtol = ((.001,.01) if out.dtype == torch.float16 else (.005,.05))
    return {'attention_correctness': 'PASS' if finite.all().item() and torch.allclose(out.float(),expected,atol=atol,rtol=rtol) else 'FAIL',
            'nonfinite': int((~finite).sum().item()),
            'per_head_max_abs': [x if x < float('inf') else 'NONFINITE' for x in head_errors],
            'atol': atol, 'rtol': rtol}


def append_expected(k, v, new_k, new_v, slots):
    """Expected cache starts as a pre-call clone; never read actual post-call cache."""
    for token, slot in enumerate(slots):
        if slot >= 0:
            k[slot//PAGE,:,slot%PAGE,:] = new_k[token]
            v[slot//PAGE,:,slot%PAGE,:new_v.shape[-1]] = new_v[token]


class Native:
    def __init__(self):
        check_error(lib.infinirtSetDevice(1,0))
        self.handle = infiniopHandle_t()
        check_error(lib.infiniopCreateHandle(C.byref(self.handle)))
        self.ops = []

    def operation(self, name, tensors):
        descriptors = []
        for t in tensors:
            d = infiniopTensorDescriptor_t()
            dtype = {torch.float16:12,torch.bfloat16:19,torch.float32:13,torch.int32:5,torch.int64:6}[t.dtype]
            check_error(lib.infiniopCreateTensorDescriptor(C.byref(d),t.ndim,(C.c_uint64*t.ndim)(*t.shape),(C.c_int64*t.ndim)(*t.stride()),dtype))
            descriptors.append(d)
        op = infiniopOperatorDescriptor_t()
        check_error(getattr(lib,'infiniopCreate'+name+'Descriptor')(self.handle,C.byref(op),*descriptors,*([None,128**-.5] if name=='PagedAttention' else [])))
        size = C.c_uint64()
        check_error(getattr(lib,'infiniopGet'+name+'WorkspaceSize')(op,C.byref(size)))
        ws = torch.empty(max(1,size.value),device='cuda',dtype=torch.uint8)
        def call():
            check_error(getattr(lib,'infiniop'+name)(op,ws.data_ptr(),size.value,*[t.data_ptr() for t in tensors],*([None] if name=='PagedAttention' else []),torch.cuda.current_stream().cuda_stream))
        self.ops.append((name,op,descriptors,ws,tensors))
        return call

    def close(self):
        for name,op,ds,_,_ in self.ops:
            check_error(getattr(lib,'infiniopDestroy'+name+'Descriptor')(op))
            for d in ds: check_error(lib.infiniopDestroyTensorDescriptor(d))
        self.ops.clear()
        check_error(lib.infiniopDestroyHandle(self.handle))
