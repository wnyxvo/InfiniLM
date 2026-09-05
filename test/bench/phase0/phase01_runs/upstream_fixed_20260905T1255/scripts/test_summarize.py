"""Regression checks for archive immutability and invalid-result exclusion."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SummaryTest(unittest.TestCase):
    def test_missing_input_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),
                '--run-dir',tmp,'--output',tmp+'/summary'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('INCOMPLETE',result.stderr)
            self.assertFalse(Path(tmp,'summary').exists())

    def test_raw_immutable_and_graph_pass_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text('{"historical_hash":"UNKNOWN"}')
            with (root/'raw.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['case','dtype','requested_variant','actual_execution_mode','correctness','latency_us'])
                w.writerow(['same','bf16','gqa','graph_batch','PASS',1])
                w.writerow(['same','bf16','gqa','eager','FAIL',2])
                w.writerow(['same','bf16','cta','eager','PASS',3])
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            command=[sys.executable,str(Path(__file__).with_name('summarize.py')),
                '--run-dir',tmp,'--raw',str(root/'raw.csv'),'--output',str(root/'summary')]
            subprocess.run(command,check=True,capture_output=True)
            for name,digest in before.items():self.assertEqual(digest,hashlib.sha256((root/name).read_bytes()).hexdigest())
            summary=json.loads((root/'summary/summary.json').read_text())
            self.assertEqual(summary['excluded_latency_rows'],{'UNVERIFIED_LEGACY_GRAPH':1,'FAIL':1})
            rows=list(csv.DictReader((root/'summary/latency_summary.csv').open()))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['variant'],'cta')
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
