# GVR Top-K Per-Phase Timing Profiling

Minimal-invasion instrumentation to measure the wall-clock time breakdown of the four phases (Guess / Verify / Collect / Refine) in the GVR heuristic Top-K kernel.

## How It Works

The patched `heuristic_topk_timed.cuh` adds `clock64()` calls at each phase boundary, guarded by `#ifdef GVR_PHASE_TIMING`. The 5 timestamps are written into the **tail of `outputValues`** (positions `[topK..topK+9]`), which is the existing write-only scratch buffer — no additional HBM allocation is needed.

```
outputValues[0..2047]   = normal Top-K values output (ignored by DSA pipeline)
outputValues[2048..2057] = 5 × int64 phase timestamps (only with GVR_PHASE_TIMING)
```

## Files

```
gvr_phase_timing/
├── README.md                         # This file
├── heuristic_topk_timed.cuh          # Patched kernel with clock64() instrumentation
├── parse_phase_timing.py             # Python script: run kernel + read timestamps
├── ablation_preidx_experiment.py     # Prediction quality ablation (existing)
├── run_phase_timing.sh               # One-click experiment runner
└── results/                          # Output directory (created by runner)
```

## Usage

### Step 1: Apply the Patch

Replace the production kernel with the timed version **for profiling only**:

```bash
# Backup original
cp cpp/tensorrt_llm/kernels/heuristic_topk.cuh \
   cpp/tensorrt_llm/kernels/heuristic_topk.cuh.bak

# Apply timed version
cp tensorrt_llm/profiling/gvr_phase_timing/heuristic_topk_timed.cuh \
   cpp/tensorrt_llm/kernels/heuristic_topk.cuh
```

### Step 2: Rebuild with Timing Enabled

```bash
# Add the define to CMake
cmake -DCMAKE_CUDA_FLAGS="-DGVR_PHASE_TIMING" ...
# Or add to the specific TU's compile flags:
# target_compile_definitions(... PRIVATE GVR_PHASE_TIMING)

make -j$(nproc) tensorrt_llm
```

**Important**: The `GVR_PHASE_TIMING` macro adds:
- 5 `clock64()` calls (~5 cycles each, negligible)
- 4 `__syncthreads()` at phase boundaries (these ARE measurable — ~2–4 µs total overhead)
- 1 global memory write (40 bytes at kernel end)

Total overhead: **~3–5 µs per kernel invocation** (~10–15% of typical kernel time). This is acceptable for profiling but should NOT be used in production.

### Step 3: Run the Experiment

```bash
# Requires SWE-Bench-64K decode logits data
bash tensorrt_llm/profiling/gvr_phase_timing/run_phase_timing.sh \
    /path/to/SWE_Bench_64K_decode_logits
```

Or run the Python script directly:

```bash
python tensorrt_llm/profiling/gvr_phase_timing/parse_phase_timing.py \
    --data_dir /path/to/SWE_Bench_64K_decode_logits \
    --layers 0 1 20 21 22 40 41 42 60 \
    --sm_freq_ghz 2.1 \
    --output phase_timing_results.json
```

### Step 4: Restore Production Kernel

```bash
cp cpp/tensorrt_llm/kernels/heuristic_topk.cuh.bak \
   cpp/tensorrt_llm/kernels/heuristic_topk.cuh
# Rebuild without GVR_PHASE_TIMING
cmake ... && make -j$(nproc) tensorrt_llm
```

## Expected Output

```
================================================================
SUMMARY: Per-layer average phase breakdown
================================================================
 Layer  Total(µs)   P1(Guess)   P2(Verify)  P3(Collect)  P4(Refine)
--------------------------------------------------------------------------------
L   0       27.2     2.8 (10.3%)    14.1 (51.8%)     6.5 (23.9%)     3.8 (14.0%)
L   1       26.1     2.7 (10.3%)    13.5 (51.7%)     6.2 (23.8%)     3.7 (14.2%)
L  20       23.9     2.5 (10.5%)     8.6 (36.0%)     7.8 (32.6%)     5.0 (20.9%)
L  21       21.8     2.4 (11.0%)     7.4 (33.9%)     7.2 (33.0%)     4.8 (22.0%)
...
```

**Key patterns to look for:**
- **Phase 2 dominates** for low-correlation layers (L0, L1): ~52% — more iterations needed
- **Phase 2 shrinks** for high-correlation layers (L20-60): ~34% — fewer iterations
- **Phase 3 is constant-ish**: ~24-33% — depends on N, not on prediction quality
- **Phase 4 is small and stable**: ~14-22% — pure SMEM work, independent of N

## Interpreting Results

The phase breakdown reveals:

| Phase | Depends on | Key insight |
|-------|-----------|-------------|
| P1 (Guess) | M=2048 (constant) | Fixed overhead, ~10% |
| P2 (Verify) | Prediction quality α | **Main variable**: 1-2 iters at high α, 4-6 at low α |
| P3 (Collect) | N (sequence length) | One full N-scan, ballot-free design |
| P4 (Refine) | Candidate count C ≤ 6144 | Pure SMEM, fast |

## Overhead Analysis

The `__syncthreads()` barriers at phase boundaries add ~2-4 µs total. To validate that this overhead doesn't distort the measurement:

1. Compare total kernel time (with timing) vs nsys-measured kernel time (without timing)
2. If difference < 15%, the phase ratios are trustworthy
3. The absolute µs values should be adjusted by subtracting the estimated sync overhead

## Related Experiments

- **Prediction quality ablation**: `ablation_preidx_experiment.py` — measures speedup vs preIdx quality (random / prev-step Top-K)
- **Per-layer latency profiling**: Use `nsys profile` with the production kernel (no timing patch needed)
