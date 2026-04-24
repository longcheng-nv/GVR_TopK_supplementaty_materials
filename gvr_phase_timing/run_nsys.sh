#!/bin/bash
#
export SWE_BENCH_PATH=/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits
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
