#!/bin/bash
#
export SWE_BENCH_PATH="${SWE_BENCH_PATH:-$1}"
if [ -z "${SWE_BENCH_PATH}" ]; then
    echo "Usage: bash run_nsys.sh /path/to/SWE_Bench_64K_decode_logits"
    echo "   or: SWE_BENCH_PATH=/path/to/SWE_Bench_64K_decode_logits bash run_nsys.sh"
    exit 1
fi
nsys profile --trace=cuda,nvtx --force-overwrite=true -o ablation \
    python ablation_preidx_experiment.py profile \
        --data_dir $SWE_BENCH_PATH \
        --layers 0 1 20 21 22 40 41 42 60 \
        --warmup 3

# export NVTX GPU projection trace CSV (maps NVTX ranges to GPU kernel durations)
nsys stats --report nvtx_gpu_proj_trace --format csv --force-export=true -o ablation_nvtx ablation.nsys-rep

# merge nsys timing + overlap data, print summary sheet
python ablation_preidx_experiment.py parse \
    --nsys_csv ablation_nvtx_nvtx_gpu_proj_trace.csv \
    --overlap_json ablation_overlap.json \
    --output ablation_preidx_results.json
