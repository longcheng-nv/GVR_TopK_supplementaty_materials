#!/bin/bash
#
#
v_no=$1
export SWE_BENCH_PATH="${SWE_BENCH_PATH:-$2}"
if [ -z "${SWE_BENCH_PATH}" ]; then
    echo "Usage: bash run_phase_timing.sh <version_tag> /path/to/SWE_Bench_64K_decode_logits"
    echo "   or: SWE_BENCH_PATH=/path/to/SWE_Bench_64K_decode_logits bash run_phase_timing.sh <version_tag>"
    exit 1
fi
export SM_FREQ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)

echo "SM clock: ${SM_FREQ} MHz"
python parse_phase_timing.py \
    --data_dir $SWE_BENCH_PATH \
    --layers 0 1 20 21 22 40 41 42 60 \
    --stride 128 \
    --warmup 3 \
    --sm_freq_ghz $(python3 -c "print(${SM_FREQ}/1000)") \
    --output results/phase_timing_results_v{v_no}.json
