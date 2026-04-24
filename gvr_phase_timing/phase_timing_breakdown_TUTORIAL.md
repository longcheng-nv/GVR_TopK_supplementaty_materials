# Phase Timing Breakdown Patch — Tutorial

## What This Patch Does

Instruments the heuristic TopK kernel with `clock64()` timestamps at each phase boundary.
After the kernel runs, 5 `long long` timestamps are written to `outputValues[topK .. topK+9]`
(10 float slots, 2 per timestamp):

| Index | Meaning |
|-------|---------|
| `ts[0]` | Phase 1 start |
| `ts[1]` | Phase 1 end / Phase 2 start |
| `ts[2]` | Phase 2 end / Phase 3 start |
| `ts[3]` | Phase 3 end / Phase 4 start |
| `ts[4]` | Kernel end |

Phase durations (in GPU clock cycles):
- **Phase 1** (heuristic sampling): `ts[1] - ts[0]`
- **Phase 2** (threshold search): `ts[2] - ts[1]`
- **Phase 3** (candidate collect): `ts[3] - ts[2]`
- **Phase 4** (histogram selection): `ts[4] - ts[3]`
- **Total**: `ts[4] - ts[0]`

## Applying the Patch

```bash
cd /path/to/TensorRT-LLM

# Dry-run (check for conflicts without modifying files)
git apply --check phase_timing_breakdown.patch

# Apply
git apply phase_timing_breakdown.patch
```

## Reverting the Patch

```bash
git apply -R phase_timing_breakdown.patch
```

## Build

Build TensorRT-LLM as usual. The instrumentation is gated behind `#define GVR_PHASE_TIMING`,
which the patch adds only in `heuristicTopKDecode.cu`. The standalone `launchHeuristicTopK`
wrapper in the header is unaffected unless you define the macro yourself.

## Reading Timestamps from Host

After kernel execution, copy the timing data from the tail of the `scratchValues` buffer:

```cpp
// scratchValues layout per row: [topK floats] [5 x long long = 10 floats]
long long host_ts[5];
cudaMemcpy(host_ts,
           scratchValues + rowIdx * topK + topK,  // past the topK values
           5 * sizeof(long long),
           cudaMemcpyDeviceToHost);

// Print phase durations in clock cycles
printf("Phase 1 (heuristic):  %lld cycles\n", host_ts[1] - host_ts[0]);
printf("Phase 2 (threshold):  %lld cycles\n", host_ts[2] - host_ts[1]);
printf("Phase 3 (collect):    %lld cycles\n", host_ts[3] - host_ts[2]);
printf("Phase 4 (histogram):  %lld cycles\n", host_ts[4] - host_ts[3]);
printf("Total:                %lld cycles\n", host_ts[4] - host_ts[0]);
```

To convert cycles to microseconds, divide by the GPU clock rate:

```cpp
// Query GPU clock rate (kHz)
int clock_khz;
cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
double us = (double)(host_ts[4] - host_ts[0]) / (double)clock_khz * 1000.0;
```

## Important Notes

- The `scratchValues` buffer must have room for at least `topK + 10` floats per row.
  In TensorRT-LLM's DSA path this is already the case since the buffer is sized generously.
- Timing adds `__syncthreads()` barriers at phase boundaries, which may slightly inflate
  measured times vs production. Use for relative phase comparison, not absolute latency.
- Remove the patch before submitting production code or running benchmarks.
