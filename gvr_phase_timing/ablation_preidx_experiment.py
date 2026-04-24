#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ablation experiment: Prediction Signal Quality for GVR Top-K.

Measures kernel latency and Top-K overlap for four preIdx variants:
  (a) No preIdx     — radix-select fallback (baseline)
  (b) Random indices — M=2048 uniform random in [0, N)
  (c) Prev-step Top-K on high-correlation layers (L20-60)
  (d) Prev-step Top-K on low-correlation layers (L0-1)

Usage (on B200 server with TensorRT-LLM installed):
  python ablation_preidx_experiment.py \
      --data_dir /path/to/SWE_Bench_64K_decode_logits \
      --layers 0 1 20 21 22 40 41 42 60 \
      --warmup 5 --repeats 20

The data_dir should contain Layer_{L}_pd.npy files from real DeepSeek-V3.2
SWE-Bench-64K decode capture.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import tensorrt_llm  # noqa: F401  — loads custom CUDA ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K = 2048
HEURISTIC_SIZE = 2048
L2_FLUSH_BYTES = 128 * 1024 * 1024  # 128 MB > B200 L2 (64 MB)


# ---------------------------------------------------------------------------
# L2 cache flush utility
# ---------------------------------------------------------------------------
def make_flush_buf(device="cuda"):
    return torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Data loading (real DeepSeek-V3.2 SWE-Bench-64K decode logits)
# ---------------------------------------------------------------------------
def load_layer_logits(data_dir, layer_no):
    path = os.path.join(data_dir, f"Layer_{layer_no}_pd.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    arr = np.load(path)
    return torch.from_numpy(arr).to(device="cuda", dtype=torch.float32)


def get_valid_length(step_id, total_steps, max_cols):
    """Compute valid logits length for a given decode step.

    Data layout: (total_steps, max_cols) where total_steps=2025, max_cols=70690.
    Step 0 has valid length = max_cols - (total_steps - 1) = 68666.
    Step s has valid length = max_cols - (total_steps - 1 - s) = 68666 + s.
    Last step (2024) has valid length = max_cols = 70690.
    """
    return max_cols - (total_steps - 1 - step_id)


def get_consecutive_pair(lscore, row_id):
    """Extract (logits_curr, prev_topk_indices, valid_N) for decode step row_id.

    Uses the FULL row (no slicing) to avoid out-of-bounds kernel reads.
    The caller passes valid_N via seq_lens so the kernel limits its scan range.
    Invalid positions (>= valid_N) are masked to -inf.
    """
    total_rows = lscore.shape[0]   # 2025
    max_cols = lscore.shape[1]     # 70690
    if row_id <= 0 or row_id >= total_rows - 1:
        raise ValueError(f"row_id={row_id} out of range [1, {total_rows - 2}]")

    N_prev = get_valid_length(row_id, total_rows, max_cols)
    logits_prev = lscore[row_id, :].unsqueeze(0).clone()
    logits_prev[:, N_prev:] = float("-inf")

    _, prev_topk = torch.topk(logits_prev, TOP_K, largest=True, sorted=False)

    N_curr = get_valid_length(row_id + 1, total_rows, max_cols)
    logits_curr = lscore[row_id + 1, :].unsqueeze(0).clone()
    logits_curr[:, N_curr:] = float("-inf")

    # Pass raw prev_topk indices (no +1 shift).
    # The kernel internally applies preIdxOffset = +1 in heuristicTopKMultiRowKernel.
    prev_topk_raw = prev_topk.to(torch.int32)

    return logits_curr, prev_topk_raw, N_curr


# ---------------------------------------------------------------------------
# Overlap measurement
# ---------------------------------------------------------------------------
def compute_overlap(pre_idx, true_topk_indices):
    """Fraction of pre_idx that appear in the true Top-K set."""
    pre_set = set(pre_idx.cpu().numpy().ravel().tolist())
    true_set = set(true_topk_indices.cpu().numpy().ravel().tolist())
    if len(true_set) == 0:
        return 0.0
    return len(pre_set & true_set) / len(true_set)


# ---------------------------------------------------------------------------
# Kernel launch with L2 flush (nsys-compatible, no Python-side timing)
# ---------------------------------------------------------------------------
def launch_kernel(logits, seq_lens, indices, next_n, topk,
                  pre_idx, scratch, flush_buf, tag=""):
    """Launch a single kernel with L2 flush, tagged with NVTX for nsys.

    Timing is NOT measured in Python. Run the script under:
        nsys profile --trace=cuda,nvtx --output=ablation python script.py
    Then extract per-kernel GPU durations from the nsys report.
    """
    flush_l2(flush_buf)
    logits_copy = logits.clone()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(tag)
    if pre_idx is not None:
        torch.ops.trtllm.indexer_topk_decode(
            logits_copy, seq_lens, indices, next_n, topk, pre_idx, scratch)
    else:
        torch.ops.trtllm.indexer_topk_decode(
            logits_copy, seq_lens, indices, next_n, topk)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Run one layer's ablation across all decode steps
# ---------------------------------------------------------------------------
def run_layer_ablation(data_dir, layer_no, step_ids, warmup, repeats=1):
    print(f"\n{'='*60}")
    print(f"Layer {layer_no}")
    print(f"{'='*60}")

    lscore = load_layer_logits(data_dir, layer_no)
    total_rows = lscore.shape[0]
    flush_buf = make_flush_buf()

    results = []

    max_cols = lscore.shape[1]  # 70690

    for step_id in step_ids:
        if step_id <= 0 or step_id >= total_rows - 1:
            continue

        logits_curr, prev_topk, N = get_consecutive_pair(lscore, step_id)

        seq_lens = torch.tensor([N], dtype=torch.int32, device="cuda")
        indices = torch.empty((1, TOP_K), dtype=torch.int32, device="cuda")
        scratch = torch.empty((1, TOP_K), dtype=torch.float32, device="cuda")

        _, true_topk = torch.topk(logits_curr[:, :N], TOP_K, largest=True, sorted=False)

        # (a) No preIdx — radix fallback
        tag_a = f"L{layer_no}/step{step_id}/radix"
        launch_kernel(logits_curr, seq_lens, indices, 1, TOP_K,
                      None, None, flush_buf, tag=tag_a)

        # (b) Random indices
        random_pre_idx = torch.randint(
            0, N, (1, HEURISTIC_SIZE), device="cuda", dtype=torch.int32)
        overlap_rand = compute_overlap(
            torch.clamp(random_pre_idx + 1, max=N - 1), true_topk)
        tag_b = f"L{layer_no}/step{step_id}/random"
        launch_kernel(logits_curr, seq_lens, indices, 1, TOP_K,
                      random_pre_idx, scratch, flush_buf, tag=tag_b)

        # (c/d) Previous-step Top-K (raw indices; kernel applies +1 internally)
        overlap_prev = compute_overlap(
            torch.clamp(prev_topk + 1, max=N - 1), true_topk)
        tag_c = f"L{layer_no}/step{step_id}/prev_topk"
        launch_kernel(logits_curr, seq_lens, indices, 1, TOP_K,
                      prev_topk, scratch, flush_buf, tag=tag_c)

        row = {
            "layer": layer_no,
            "step": step_id,
            "N": N,
            "random_overlap": overlap_rand,
            "prev_topk_overlap": overlap_prev,
        }
        results.append(row)
        print(f"  step={step_id:4d}  N={N:6d}  "
              f"α_rand={overlap_rand:.3f}  α_prev={overlap_prev:.3f}")

    del lscore, flush_buf
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_nsys_csv(csv_path, overlap_json_path):
    """Parse nsys cuda_gpu_trace CSV and merge with overlap data.

    Generate CSV with:
        nsys stats --report cuda_gpu_trace --format csv \
            -o ablation_gpu_trace ablation.nsys-rep
    The CSV file will be at ablation_gpu_trace_cuda_gpu_trace.csv
    """
    import csv

    with open(overlap_json_path) as f:
        overlap_data = json.load(f)

    overlap_map = {}
    for r in overlap_data:
        key = (r["layer"], r["step"])
        overlap_map[key] = r

    kernel_times = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support two CSV formats:
            #  1. nvtx_gpu_proj_trace: NVTX name in "Name", duration in
            #     "Projected Duration (ns)"
            #  2. cuda_gpu_trace with NVTX column: NVTX in "NVTX Range"
            nvtx = row.get("NVTX Range", row.get("NvtxRange", ""))
            if not nvtx:
                # Try nvtx_gpu_proj_trace format: Name column holds NVTX
                nvtx = row.get("Name", "")
            dur_ns = float(row.get("Projected Duration (ns)",
                           row.get("Duration (ns)",
                           row.get("Duration", 0))))

            # Strip leading ':' from NVTX range stack names
            nvtx = nvtx.lstrip(":")

            if not nvtx or "/" not in nvtx:
                continue

            parts = nvtx.split("/")
            if len(parts) != 3:
                continue
            layer_str, step_str, variant = parts
            layer = int(layer_str.replace("L", ""))
            step = int(step_str.replace("step", ""))

            key = (layer, step, variant)
            # Sum durations: an NVTX range may project onto multiple GPU ops
            kernel_times[key] = kernel_times.get(key, 0) + dur_ns

    all_results = []
    for (layer, step), ov in sorted(overlap_map.items()):
        radix_ns = kernel_times.get((layer, step, "radix"), 0)
        random_ns = kernel_times.get((layer, step, "random"), 0)
        prev_ns = kernel_times.get((layer, step, "prev_topk"), 0)
        row = {
            "layer": layer, "step": step, "N": ov["N"],
            "no_preidx_mean_ns": radix_ns,
            "random_mean_ns": random_ns, "random_overlap": ov["random_overlap"],
            "prev_topk_mean_ns": prev_ns, "prev_topk_overlap": ov["prev_topk_overlap"],
        }
        if radix_ns > 0:
            row["random_speedup"] = radix_ns / random_ns if random_ns > 0 else 0
            row["prev_topk_speedup"] = radix_ns / prev_ns if prev_ns > 0 else 0
        else:
            row["random_speedup"] = 0
            row["prev_topk_speedup"] = 0
        all_results.append(row)

    return all_results


def print_summary_tables(all_results, layers):
    """Print TABLE 1/2/3 from merged results."""

    print("\n" + "=" * 80)
    print("TABLE 1: Per-layer summary (avg over sampled decode steps)")
    print("=" * 80)
    header = (f"{'Layer':>6s}  {'#Steps':>6s}  "
              f"{'Radix(us)':>10s}  "
              f"{'Random(us)':>10s}  {'Rand-α':>7s}  {'Rand-x':>7s}  "
              f"{'PrevTK(us)':>10s}  {'Prev-α':>7s}  {'Prev-x':>7s}  "
              f"{'Prev-x min':>10s}  {'Prev-x max':>10s}")
    print(header)
    for layer_no in layers:
        rows = [r for r in all_results if r["layer"] == layer_no]
        if not rows:
            continue
        n = len(rows)
        radix = np.mean([r["no_preidx_mean_ns"] for r in rows]) / 1e3
        rand_us = np.mean([r["random_mean_ns"] for r in rows]) / 1e3
        rand_a = np.mean([r["random_overlap"] for r in rows])
        rand_x = np.mean([r["random_speedup"] for r in rows])
        prev_us = np.mean([r["prev_topk_mean_ns"] for r in rows]) / 1e3
        prev_a = np.mean([r["prev_topk_overlap"] for r in rows])
        prev_x = np.mean([r["prev_topk_speedup"] for r in rows])
        prev_x_min = np.min([r["prev_topk_speedup"] for r in rows])
        prev_x_max = np.max([r["prev_topk_speedup"] for r in rows])
        print(f"L{layer_no:>4d}  {n:6d}  "
              f"{radix:10.1f}  "
              f"{rand_us:10.1f}  {rand_a:7.3f}  {rand_x:7.2f}  "
              f"{prev_us:10.1f}  {prev_a:7.3f}  {prev_x:7.2f}  "
              f"{prev_x_min:10.2f}  {prev_x_max:10.2f}")

    print("\n" + "=" * 80)
    print("TABLE 2: Per-variant summary (paper ablation table format)")
    print("=" * 80)
    variant_info = [
        ("(a) No preIdx (radix)", "no_preidx"),
        ("(b) Random indices",    "random"),
        ("(c) Prev-step Top-K",   "prev_topk"),
    ]
    print(f"{'Variant':<30s}  {'Overlap(α)':>10s}  "
          f"{'Latency(us)':>11s}  {'Speedup':>8s}  "
          f"{'Speedup min':>11s}  {'Speedup max':>11s}")
    for label, key in variant_info:
        latencies = [r[f"{key}_mean_ns"] for r in all_results]
        overlaps = [r.get(f"{key}_overlap", 0.0) for r in all_results]
        speedups = [r.get(f"{key}_speedup", 1.0) for r in all_results]
        print(f"{label:<30s}  {np.mean(overlaps):10.3f}  "
              f"{np.mean(latencies)/1e3:11.1f}  {np.mean(speedups):8.2f}  "
              f"{np.min(speedups):11.2f}  {np.max(speedups):11.2f}")

    print("\n" + "=" * 80)
    print("TABLE 3: Prev-step Top-K — high-corr layers vs low-corr layers")
    print("=" * 80)
    high_corr_layers = {20, 21, 22, 40, 41, 42, 60}
    low_corr_layers = {0, 1}
    for group_name, group_set in [("(c) High-corr L20-60", high_corr_layers),
                                   ("(d) Low-corr L0-1",    low_corr_layers)]:
        rows = [r for r in all_results if r["layer"] in group_set]
        if not rows:
            continue
        overlaps = [r["prev_topk_overlap"] for r in rows]
        latencies = [r["prev_topk_mean_ns"] for r in rows]
        speedups = [r["prev_topk_speedup"] for r in rows]
        print(f"  {group_name}:")
        print(f"    Samples:     {len(rows)}")
        print(f"    Avg overlap: {np.mean(overlaps):.3f}")
        print(f"    Avg latency: {np.mean(latencies)/1e3:.1f} us")
        print(f"    Avg speedup: {np.mean(speedups):.2f}x "
              f"(min {np.min(speedups):.2f}x, max {np.max(speedups):.2f}x)")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: Prediction Signal Quality for GVR Top-K")
    subparsers = parser.add_subparsers(dest="command")

    # --- Sub-command: profile (run kernels, collect overlap, to be profiled by nsys) ---
    p_profile = subparsers.add_parser("profile",
        help="Run kernels with NVTX tags (use with nsys profile)")
    p_profile.add_argument("--data_dir", type=str, required=True,
                           help="Path to SWE_Bench_64K_decode_logits/ directory")
    p_profile.add_argument("--layers", type=int, nargs="+",
                           default=[0, 1, 20, 21, 22, 40, 41, 42, 60])
    p_profile.add_argument("--step_stride", type=int, default=128)
    p_profile.add_argument("--warmup", type=int, default=3,
                           help="Warmup launches per variant (before profiled call)")
    p_profile.add_argument("--output", type=str, default="ablation_overlap.json",
                           help="Output JSON with overlap data (timing from nsys)")

    # --- Sub-command: parse (merge nsys CSV with overlap JSON, print tables) ---
    p_parse = subparsers.add_parser("parse",
        help="Parse nsys GPU trace CSV + overlap JSON, print summary tables")
    p_parse.add_argument("--nsys_csv", type=str, required=True,
                         help="nsys cuda_gpu_trace CSV file")
    p_parse.add_argument("--overlap_json", type=str, required=True,
                         help="Overlap JSON from 'profile' step")
    p_parse.add_argument("--layers", type=int, nargs="+",
                         default=[0, 1, 20, 21, 22, 40, 41, 42, 60])
    p_parse.add_argument("--output", type=str, default="ablation_preidx_results.json")

    args = parser.parse_args()

    if args.command == "profile":
        probe = np.load(os.path.join(args.data_dir,
                                     f"Layer_{args.layers[0]}_pd.npy"))
        total_steps, max_cols = probe.shape
        del probe
        step_ids = list(range(1, total_steps - 1, args.step_stride))
        last_valid = total_steps - 2
        if last_valid not in step_ids:
            step_ids.append(last_valid)
        N_first = get_valid_length(step_ids[0] + 1, total_steps, max_cols)
        N_last = get_valid_length(last_valid + 1, total_steps, max_cols)
        print(f"Data shape: ({total_steps}, {max_cols})")
        print(f"Sampling {len(step_ids)} decode steps (stride={args.step_stride})")
        print(f"Valid N range: {N_first} (step {step_ids[0]+1}) .. "
              f"{N_last} (step {last_valid+1})")

        all_results = []
        for layer_no in args.layers:
            layer_results = run_layer_ablation(
                args.data_dir, layer_no, step_ids, args.warmup, repeats=1)
            all_results.extend(layer_results)

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nOverlap data saved to {args.output}")
        print("\nNext steps:")
        print("  1. Run this script under nsys:")
        print(f"     nsys profile --trace=cuda,nvtx -o ablation "
              f"python {sys.argv[0]} profile --data_dir <path> ...")
        print("  2. Export GPU trace CSV:")
        print("     nsys stats --report cuda_gpu_trace --format csv "
              "-o ablation_gpu ablation.nsys-rep")
        print("  3. Parse and merge:")
        print(f"     python {sys.argv[0]} parse "
              "--nsys_csv ablation_gpu_cuda_gpu_trace.csv "
              f"--overlap_json {args.output}")

    elif args.command == "parse":
        all_results = parse_nsys_csv(args.nsys_csv, args.overlap_json)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Merged results saved to {args.output}")
        print_summary_tables(all_results, args.layers)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
