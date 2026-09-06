"""Correctness matrix for the 32-thread vector path and scalar fallbacks."""
import argparse
import os
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.correctness_support import Run, Native, PAGE


def make_tensor(shape, dtype, last_stride=1, offset=0):
    # Build an explicit strided view so storage_offset and innermost stride are
    # exercised rather than inferred from a contiguous tensor.
    stride = [0] * len(shape)
    stride[-1] = last_stride
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride)) + offset
    base = torch.randn(span, device="cuda", dtype=dtype)
    return base.as_strided(shape, stride, storage_offset=offset)


def expected_cache(cache, source, slots):
    result = cache.clone()
    for t, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        for h in range(source.shape[1]):
            result[slot // PAGE, h, slot % PAGE, : source.shape[2]] = source[t, h]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(20260906)
    run = Run(args.output)
    cases = [
        ("vector_fp16_aligned", torch.float16, 128, 128, 1, 0, True, 1),
        ("vector_bf16_aligned", torch.bfloat16, 128, 128, 1, 0, True, 1),
        ("fallback_storage_offset", torch.float16, 128, 128, 1, 1, True, 1),
        ("fallback_innermost_stride", torch.float16, 128, 128, 2, 0, True, 1),
        ("fallback_different_widths", torch.float16, 130, 96, 1, 0, True, 1),
        ("fallback_fp32_negative_slot", torch.float32, 128, 128, 1, 0, True, 1),
        ("grouped_E_hkv3", torch.float16, 128, 128, 1, 0, True, 2),
        ("grouped_E_hkv5", torch.bfloat16, 128, 128, 1, 0, True, 2),
        ("grouped_F_hkv1", torch.float16, 128, 128, 1, 0, True, 4),
        ("grouped_F_hkv5", torch.bfloat16, 128, 128, 1, 0, True, 4),
    ]
    try:
        for name, dtype, dk, dv, last_stride, offset, request_vector, grouped in cases:
            os.environ["INFINIOP_PAGED_CACHING_THREADS"] = "64" if grouped == 2 else ("128" if grouped == 4 else "32")
            if grouped > 1: os.environ["INFINIOP_PAGED_CACHING_HEADS_PER_CTA"] = str(grouped)
            else: os.environ.pop("INFINIOP_PAGED_CACHING_HEADS_PER_CTA", None)
            if request_vector:
                os.environ["INFINIOP_PAGED_CACHING_VECTOR"] = "1"
            else:
                os.environ.pop("INFINIOP_PAGED_CACHING_VECTOR", None)
            tokens, heads, blocks = 5, (3 if "hkv3" in name else (1 if "hkv1" in name else (5 if "hkv5" in name else 8))), 3
            k = make_tensor((tokens, heads, dk), dtype, last_stride, offset)
            v = make_tensor((tokens, heads, dv), dtype, last_stride, offset)
            kc = make_tensor((blocks, heads, PAGE, dk), dtype, last_stride, offset)
            vc = make_tensor((blocks, heads, PAGE, dv), dtype, last_stride, offset)
            slots = torch.tensor([0, 255, 256, 511, -1], device="cuda", dtype=torch.int64)
            # Save the original cache before any measured call; all expected
            # states are derived from this immutable pre-call snapshot.
            base_k, base_v = kc.clone(), vc.clone()
            ek = expected_cache(base_k, k, slots)
            ev = expected_cache(base_v, v, slots)
            native = Native()
            call = native.operation("PagedCaching", [kc, vc, k, v, slots])
            call(); torch.cuda.synchronize()
            eager_pass = bool(torch.equal(kc, ek) and torch.equal(vc, ev))
            # Capture from the original state, then restore it before replay.
            kc.copy_(base_k); vc.copy_(base_v)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            torch.cuda.synchronize()
            kc.copy_(base_k); vc.copy_(base_v)
            skipped_replay_detected = bool(torch.equal(kc, base_k) and torch.equal(vc, base_v) and not torch.equal(kc, ek))
            graph.replay(); torch.cuda.synchronize()
            graph_pass = bool(torch.equal(kc, ek) and torch.equal(vc, ev))
            # Update source tensors and legal metadata on the same captured graph.
            k.copy_(make_tensor((tokens, heads, dk), dtype, last_stride, offset))
            v.copy_(make_tensor((tokens, heads, dv), dtype, last_stride, offset))
            slots.copy_(torch.tensor([2, 254, 257, 510, -1], device="cuda", dtype=torch.int64))
            update_ek = expected_cache(base_k, k, slots); update_ev = expected_cache(base_v, v, slots)
            kc.copy_(base_k); vc.copy_(base_v); graph.replay(); torch.cuda.synchronize()
            graph_update_pass = bool(torch.equal(kc, update_ek) and torch.equal(vc, update_ev))
            run.add(case=name, dtype=str(dtype), head_size=dk, v_head_size=dv,
                    innermost_stride=last_stride, storage_offset=offset,
                    vector_requested=request_vector, eager_correctness="PASS" if eager_pass else "FAIL",
                    graph_correctness="PASS" if graph_pass else "FAIL",
                    graph_update_correctness="PASS" if graph_update_pass else "FAIL",
                    skip_replay_negative_control="PASS" if skipped_replay_detected else "FAIL",
                    negative_slot_preserved=bool(torch.equal(kc[-1], base_k[-1])),
                    heads_per_cta=grouped,
                    result="PASS" if eager_pass and graph_pass and graph_update_pass and skipped_replay_detected else "FAIL")
            del graph
            native.close()
        return run.finish()
    except Exception as error:
        run.finish(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
