#!/bin/bash
#
#
v_no=$1
export SWE_BENCH_PATH=/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits
export SM_FREQ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)

echo "SM clock: ${SM_FREQ} MHz"
python parse_phase_timing.py \
    --data_dir $SWE_BENCH_PATH \
    --layers 0 1 20 21 22 40 41 42 60 \
    --stride 128 \
    --warmup 3 \
    --sm_freq_ghz $(python3 -c "print(${SM_FREQ}/1000)") \
    --output results/phase_timing_results_v{v_no}.json
