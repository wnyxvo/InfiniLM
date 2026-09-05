"""Run existing native attention/MHA and caching tests with captured provenance."""
import argparse
import runpy
import sys
import torch
from correctness_support import Run, CORE, SEED, configure


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    run=Run(a.output);configure('default')
    try:
        for name in ['paged_attention','paged_caching']:
            torch.manual_seed(SEED)
            script=CORE/'test/infiniop'/f'{name}.py'
            previous=sys.argv
            try:
                sys.argv=[str(script),'--nvidia']
                namespace=runpy.run_path(str(script),run_name='__main__')
            finally:
                sys.argv=previous
            run.add(case=name,mode='eager',result='PASS',
                    case_count=len(namespace['_TEST_CASES_']),
                    dtypes=namespace['_TENSOR_DTYPES'])
        return run.finish()
    except Exception as error:
        run.finish(error);raise


if __name__=='__main__':sys.exit(main())
