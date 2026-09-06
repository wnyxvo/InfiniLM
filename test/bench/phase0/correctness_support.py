"""Compatibility wrapper; canonical implementation lives in decode_ops/common."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_ops.common.correctness_support import *
