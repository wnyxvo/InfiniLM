"""Current startup check only: do not instantiate or time a model."""
import argparse
import contextlib
import importlib.util
import io
import runpy
import sys
from correctness_support import Run, LM


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    run=Run(a.output)
    sys.path.insert(0,str(LM/'python'))
    log=io.StringIO()
    try:
        previous=sys.argv
        try:
            sys.argv=[str(LM/'examples/bench.py'),'--help']
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:runpy.run_path(sys.argv[0],run_name='__main__')
                except SystemExit as exit:
                    if exit.code not in (None,0):raise
        finally:sys.argv=previous
        (run.path/'execution.log').write_text(log.getvalue())
        import transformers,infinilm
        run.add(case='bench_help',mode='startup_only',result='PASS',
                transformers=transformers.__version__,infinilm_path=infinilm.__file__,
                pybind11=str(importlib.util.find_spec('pybind11')),
                e2e='NOT_RUN: correctness gate unresolved; historical missing-dependency blocker no longer applies')
        return run.finish()
    except Exception as error:
        (run.path/'execution.log').write_text(log.getvalue())
        run.finish(error);raise


if __name__=='__main__':sys.exit(main())
