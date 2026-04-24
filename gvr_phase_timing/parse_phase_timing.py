#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse per-phase timing data from GVR Top-K kernel's scratch buffer.

When compiled with -DGVR_PHASE_TIMING, the kernel writes 5 clock64()
timestamps into outputValues[topK..topK+9] (10 floats = 5 long longs).

This script reads the scratch buffer after each kernel invocation and
computes per-phase wall-clock durations.

Usage:
    python parse_phase_timing.py \
        --data_dir /path/to/SWE_Bench_64K_decode_logits \
        --layers 0 1 20 21 22 40 41 42 60 \
        --sm_freq_ghz 2.1 \
        --output phase_timing_results.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import tensorrt_llm  # noqa: F401 — loads custom CUDA ops


TOP_K = 2048
HEURISTIC_SIZE = 2048
L2_FLUSH_BYTES = 128 * 1024 * 1024


def make_flush_buf(device="cuda"):
    return torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


def load_layer_logits(data_dir, layer_no):
    path = os.path.join(data_dir, f"Layer_{layer_no}_pd.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path)


def extract_timestamps(scratch_values, topk=TOP_K):
    """Extract 5 clock64 timestamps from scratch_values[topK:topK+10]."""
    raw = scratch_values[topk:topk + 10].cpu().numpy()
    timestamps = np.frombuffer(raw.tobytes(), dtype=np.int64)
    return timestamps


def run_gvr_and_read_timing(logits_row, prev_topk, scratch_values, indices_out,
                             kv_lens, flush_buf, warmup=3):
    """Run GVR kernel with prev-step Top-K and read phase timestamps."""
    N = logits_row.shape[0]
    device = logits_row.device

    logits_2d = logits_row.unsqueeze(0)

    for _ in range(warmup):
        flush_l2(flush_buf)
        torch.ops.trtllm.indexer_topk_decode(
            logits_2d, kv_lens, indices_out, 1, TOP_K,
            pre_idx=prev_topk, heuristic_scratch=scratch_values)
        torch.cuda.synchronize()

    flush_l2(flush_buf)
    torch.ops.trtllm.indexer_topk_decode(
        logits_2d, kv_lens, indices_out, 1, TOP_K,
        pre_idx=prev_topk, heuristic_scratch=scratch_values)
    torch.cuda.synchronize()

    timestamps = extract_timestamps(scratch_values.view(-1))
    return timestamps


def main():
    parser = argparse.ArgumentParser(description="GVR Phase Timing Parser")
    parser.add_argument("--data_dir", required=True,
                        help="Path to SWE_Bench_64K_decode_logits")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 1, 20, 21, 22, 40, 41, 42, 60])
    parser.add_argument("--stride", type=int, default=128,
                        help="Decode step sampling stride")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sm_freq_ghz", type=float, default=2.1,
                        help="SM clock frequency in GHz (from nvidia-smi -q)")
    parser.add_argument("--output", default="phase_timing_results.json")
    args = parser.parse_args()

    ns_per_cycle = 1.0 / args.sm_freq_ghz
    device = "cuda"

    flush_buf = make_flush_buf(device)
    kv_lens = torch.tensor([1], dtype=torch.int32, device=device)
    indices_out = torch.zeros(1, TOP_K, dtype=torch.int32, device=device)
    scratch_values = torch.zeros(1, TOP_K + 10, dtype=torch.float32, device=device)

    phase_names = ["Phase 1 (Guess)", "Phase 2 (Verify)",
                   "Phase 3 (Collect)", "Phase 4 (Refine)"]
    results = []

    for layer in args.layers:
        print(f"\n{'='*60}\nLayer {layer}\n{'='*60}")
        all_logits = load_layer_logits(args.data_dir, layer)
        num_steps, N_max = all_logits.shape

        sample_steps = list(range(1, num_steps - 1, args.stride))
        if num_steps - 1 not in sample_steps:
            sample_steps.append(num_steps - 1)

        for step in sample_steps:
            N = N_max - (num_steps - 1 - step)
            logits_row = torch.from_numpy(
                all_logits[step, :N].astype(np.float32)).to(device)

            prev_logits = torch.from_numpy(
                all_logits[step - 1, :N].astype(np.float32)).to(device)
            prev_topk = prev_logits.topk(TOP_K).indices.int().unsqueeze(0)

            kv_lens[0] = N

            timestamps = run_gvr_and_read_timing(
                logits_row, prev_topk, scratch_values, indices_out,
                kv_lens, flush_buf, warmup=args.warmup)

            if timestamps[4] <= timestamps[0]:
                print(f"  step={step:>5d} N={N:>6d}  WARNING: invalid timestamps")
                continue

            durations_cycles = [timestamps[i+1] - timestamps[i] for i in range(4)]
            total_cycles = timestamps[4] - timestamps[0]
            durations_us = [d * ns_per_cycle / 1000 for d in durations_cycles]
            total_us = total_cycles * ns_per_cycle / 1000
            pcts = [d / total_cycles * 100 for d in durations_cycles]

            print(f"  step={step:>5d} N={N:>6d}  total={total_us:.1f}µs  "
                  f"P1={pcts[0]:.0f}% P2={pcts[1]:.0f}% "
                  f"P3={pcts[2]:.0f}% P4={pcts[3]:.0f}%")

            results.append({
                "layer": layer, "step": step, "N": N,
                "total_us": round(total_us, 2),
                "phase1_us": round(durations_us[0], 2),
                "phase2_us": round(durations_us[1], 2),
                "phase3_us": round(durations_us[2], 2),
                "phase4_us": round(durations_us[3], 2),
                "phase1_pct": round(pcts[0], 1),
                "phase2_pct": round(pcts[1], 1),
                "phase3_pct": round(pcts[2], 1),
                "phase4_pct": round(pcts[3], 1),
            })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print(f"\n{'='*80}")
    print("SUMMARY: Per-layer average phase breakdown")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'Total(µs)':>10} {'P1(Guess)':>12} {'P2(Verify)':>12} "
          f"{'P3(Collect)':>12} {'P4(Refine)':>12}")
    print("-" * 80)

    for layer in args.layers:
        layer_data = [r for r in results if r["layer"] == layer]
        if not layer_data:
            continue
        avg = lambda key: np.mean([r[key] for r in layer_data])
        print(f"L{layer:>4} {avg('total_us'):>10.1f} "
              f"{avg('phase1_us'):>7.1f} ({avg('phase1_pct'):>4.1f}%) "
              f"{avg('phase2_us'):>7.1f} ({avg('phase2_pct'):>4.1f}%) "
              f"{avg('phase3_us'):>7.1f} ({avg('phase3_pct'):>4.1f}%) "
              f"{avg('phase4_us'):>7.1f} ({avg('phase4_pct'):>4.1f}%)")


if __name__ == "__main__":
    main()
