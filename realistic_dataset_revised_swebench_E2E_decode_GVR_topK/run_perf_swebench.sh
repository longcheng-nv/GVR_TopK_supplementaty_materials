#!/bin/bash
set -ex
# A/B Benchmark: heuristic TopK vs original radix-sort TopK
# Using real SWE-bench prompts (variable ISL ~52K-69K) for realistic decode
# performance measurement.
#
# Design:
#   - BS=1: min-latency, TopK indexer is largest fraction of E2E decode time
#   - Real SWE-bench prompts: variable ISL (52K-69K), realistic token distribution
#   - OSL=512: short output to focus on decode TPOT, reduce MTP noise
#   - MTP=3: speculative decoding
#   - 5 prompts × NUM_REPS repetitions: statistical averaging
#   - Interleaved A/B: each rep runs both variants back-to-back
#
# Usage:
#   bash run_perf_swebench.sh                                           # default: swe_bench_64k, all prompts
#   bash run_perf_swebench.sh --input swe_bench_100k.jsonl              # use 100K dataset
#   bash run_perf_swebench.sh --entry 2                                 # entry #2 only, 10 requests
#   bash run_perf_swebench.sh --entry 2 --num-requests 20               # entry #2, 20 requests
#   bash run_perf_swebench.sh --input swe_bench_100k.jsonl --entry 1
#   bash run_perf_swebench.sh --dataset_osl 512 --MTP 3

# --- Parse arguments ---
SWEBENCH_JSONL=""
ENTRY_ID=""
NUM_REQUESTS=""
ARG_DATASET_OSL=""
ARG_MTP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)        SWEBENCH_JSONL="$2"; shift 2 ;;
        --entry)        ENTRY_ID="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --dataset_osl)  ARG_DATASET_OSL="$2"; shift 2 ;;
        --MTP)          ARG_MTP="$2"; shift 2 ;;
        *)
            # Backward compat: positional args (ENTRY_ID [NUM_REQUESTS])
            if [ -z "${ENTRY_ID}" ]; then ENTRY_ID="$1"
            elif [ -z "${NUM_REQUESTS}" ]; then NUM_REQUESTS="$1"
            fi
            shift ;;
    esac
done

model_card="deepseek-ai/DeepSeek-V3.2-Exp"
model_path="${MODEL_PATH:-}"
if [ -z "${model_path}" ]; then
    echo "ERROR: Please set MODEL_PATH to your local DeepSeek-V3.2-Exp-FP4-v2 model directory."
    exit 1
fi
max_batch_size=1
kv_fraction=0.8

dataset_osl=${ARG_DATASET_OSL:-2025}
ep=8
MTP=${ARG_MTP:-1}
max_num_tokens=8192            # chunked prefill: each chunk <= this size
NUM_REPS=3                     # repeat each A/B pair this many times

export PATH=${HOME}/.local/bin:${PATH}
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4

# Kill any leftover GPU processes and wait for memory to be reclaimed
gpu_cleanup() {
    # Kill stale python/trtllm processes on all GPUs (ignore errors)
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    if [ -n "${pids}" ]; then
        echo "[cleanup] Killing leftover GPU processes: ${pids}"
        echo "${pids}" | xargs -r kill -9 2>/dev/null || true
        sleep 5   # wait for driver to reclaim memory
    fi
    # Verify GPU memory is free
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

# --- Generate tokenized dataset from SWE-bench prompts ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="${SCRIPT_DIR}/../longseqtasks"

# Default to swe_bench_64k.jsonl if not specified
if [ -z "${SWEBENCH_JSONL}" ]; then
    SWEBENCH_JSONL="swe_bench_64k.jsonl"
fi
# Resolve to absolute path: accept bare filename (looked up in bundled longseqtasks/) or full path
if [ -f "${SWEBENCH_JSONL}" ]; then
    SWEBENCH_INPUT="$(realpath "${SWEBENCH_JSONL}")"
elif [ -f "${TASKS_DIR}/${SWEBENCH_JSONL}" ]; then
    SWEBENCH_INPUT="${TASKS_DIR}/${SWEBENCH_JSONL}"
else
    echo "ERROR: Cannot find input dataset: ${SWEBENCH_JSONL}"
    echo "  Looked in: ./ and ${TASKS_DIR}/"
    exit 1
fi

# Derive dataset name from input filename, encoding OSL to avoid stale cache
# e.g., swe_bench_100k.jsonl + osl=1024 -> dataset_swebench_100k_osl1024.json
input_basename=$(basename "${SWEBENCH_INPUT}" .jsonl)
isl_tag=$(echo "${input_basename}" | grep -oP '\d+k' || echo "unknown")
DATASET="${SCRIPT_DIR}/dataset_swebench_${isl_tag}_osl${dataset_osl}.json"

# Generate dataset with atomic rename to handle cross-node NFS races.
# Multiple nodes may run this concurrently on a shared filesystem where flock
# is unreliable. Strategy: each node writes to a unique temp file, then
# atomically renames (mv) to the final path. Since all nodes produce identical
# content for the same ISL+OSL, the last rename wins — which is safe.
if [ ! -f "${DATASET}" ]; then
    TMPFILE="${DATASET}.tmp.$(hostname -s).$$"
    MAX_ISL=$(python3 ${SCRIPT_DIR}/prepare_swebench_dataset.py \
        --input "${SWEBENCH_INPUT}" \
        --tokenizer "${model_path}" \
        --osl ${dataset_osl} \
        --output "${TMPFILE}")
    mv -f "${TMPFILE}" "${DATASET}"
    echo "${MAX_ISL}" > "${DATASET}.max_isl"
else
    # Dataset already exists — read cached max_isl or recompute
    if [ -f "${DATASET}.max_isl" ]; then
        MAX_ISL=$(cat "${DATASET}.max_isl")
    else
        MAX_ISL=$(python3 -c "
import json
max_isl = 0
with open('${DATASET}') as f:
    for line in f:
        d = json.loads(line)
        max_isl = max(max_isl, len(d['input_ids']))
print(max_isl)
")
    fi
fi

# Compute max_seq_len: 1.25× max ISL, rounded up to nearest 1024
max_seq_len=$(python3 -c "import math; print(math.ceil(int(${MAX_ISL}) * 1.25 / 1024) * 1024)")
echo "Max ISL=${MAX_ISL}, computed max_seq_len=${max_seq_len}"

# --- Log directory setup (after isl_tag is known) ---
entry_tag=""
if [ -n "${ENTRY_ID}" ]; then
    entry_tag="_entry${ENTRY_ID}"
fi
log_dir=${SCRIPT_DIR}/tmp/ds_swebench_BS${max_batch_size}_ISL${isl_tag}_OSL${dataset_osl}_fp4_heuristic_topk_perf_CUDAGraph${entry_tag}
mkdir -p ${log_dir}
timestamp=$(date +'%m%d%H%M')
log_prefix=${log_dir}/run_swebench${entry_tag}_DEP${ep}_${max_batch_size}_${max_num_tokens}_FP4_B200_MTP${MTP}_${timestamp}
config_yml=${log_dir}/extra-llm-api-config_${timestamp}_$$.yml

# --- Select entry or use full dataset ---
if [ -n "${ENTRY_ID}" ]; then
    total_entries=$(wc -l < "${DATASET}")
    if [ "${ENTRY_ID}" -ge "${total_entries}" ] || [ "${ENTRY_ID}" -lt 0 ]; then
        echo "ERROR: ENTRY_ID=${ENTRY_ID} out of range [0, $((total_entries-1))]"
        exit 1
    fi
    BENCH_DATASET="${log_dir}/dataset_swebench_entry${ENTRY_ID}.json"
    single_line=$(sed -n "$((ENTRY_ID+1))p" "${DATASET}")
    # trtllm-bench caps num_requests at dataset line count, so replicate the
    # single entry enough times to satisfy num_prompts (default 10).
    rep_count=${NUM_REQUESTS:-10}
    : > "${BENCH_DATASET}"
    for _i in $(seq 1 ${rep_count}); do echo "${single_line}" >> "${BENCH_DATASET}"; done
    entry_isl=$(python3 -c "import json; d=json.loads(open('${BENCH_DATASET}').readline()); print(len(d['input_ids']))")
    echo "Selected entry #${ENTRY_ID}: ISL=${entry_isl}, OSL=${dataset_osl}"
else
    BENCH_DATASET="${DATASET}"
fi

dataset_lines=$(wc -l < "${BENCH_DATASET}")
if [ -n "${NUM_REQUESTS}" ]; then
    num_prompts=${NUM_REQUESTS}
elif [ -n "${ENTRY_ID}" ]; then
    num_prompts=10   # single entry: repeat 10× for statistical averaging
else
    num_prompts=${dataset_lines}   # all entries: run each once
fi
echo "Dataset: ${dataset_lines} line(s), num_requests=${num_prompts}, OSL=${dataset_osl}"

# --- Repeated A/B runs ---
for rep in $(seq 1 ${NUM_REPS}); do
    echo "===== Repetition ${rep}/${NUM_REPS} ====="
    gpu_cleanup

    # --- Run A: Original radix-sort TopK ---
    cat <<EOF > ${config_yml}
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch_size}
kv_cache_config:
    free_gpu_memory_fraction: ${kv_fraction}
    enable_block_reuse: false
    tokens_per_block: 64
    dtype: fp8
enable_chunked_prefill: true
print_iter_log: false
enable_attention_dp: false
moe_config:
    backend: TRTLLM
$(if [ "${MTP}" -gt 0 ]; then
cat <<SPEC
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
SPEC
fi)
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: false
EOF

    trtllm-bench -m ${model_card} --model_path ${model_path} throughput \
        --tp ${ep} \
        --ep ${ep} \
        --warmup 2 \
        --dataset "${BENCH_DATASET}" \
        --backend pytorch \
        --max_batch_size ${max_batch_size} \
        --max_num_tokens ${max_num_tokens} \
        --max_seq_len ${max_seq_len} \
        --kv_cache_free_gpu_mem_fraction ${kv_fraction} \
        --concurrency ${max_batch_size} \
        --extra_llm_api_options ${config_yml} \
        --num_requests ${num_prompts} \
        --streaming |& tee ${log_prefix}_rep${rep}_oriTopK.txt

    gpu_cleanup

    # --- Run B: Heuristic TopK ---
    cat <<EOF > ${config_yml}
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch_size}
kv_cache_config:
    free_gpu_memory_fraction: ${kv_fraction}
    enable_block_reuse: false
    tokens_per_block: 64
    dtype: fp8
enable_chunked_prefill: true
print_iter_log: false
enable_attention_dp: false
moe_config:
    backend: TRTLLM
$(if [ "${MTP}" -gt 0 ]; then
cat <<SPEC
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
SPEC
fi)
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: true
EOF

    trtllm-bench -m ${model_card} --model_path ${model_path} throughput \
        --tp ${ep} \
        --ep ${ep} \
        --warmup 2 \
        --dataset "${BENCH_DATASET}" \
        --backend pytorch \
        --max_batch_size ${max_batch_size} \
        --max_num_tokens ${max_num_tokens} \
        --max_seq_len ${max_seq_len} \
        --kv_cache_free_gpu_mem_fraction ${kv_fraction} \
        --concurrency ${max_batch_size} \
        --extra_llm_api_options ${config_yml} \
        --num_requests ${num_prompts} \
        --streaming |& tee ${log_prefix}_rep${rep}_heuristicTopK.txt
done

# --- Summary across all repetitions ---
echo ""
echo "============================================"
echo "=== Results Summary (${NUM_REPS} repetitions) ==="
echo "============================================"
entry_info=""
if [ -n "${ENTRY_ID}" ]; then entry_info=", entry=${ENTRY_ID}"; fi
echo "Config: BS=${max_batch_size}, ISL=${isl_tag}(max=${MAX_ISL}), OSL=${dataset_osl}, MTP=${MTP}, max_seq_len=${max_seq_len}, requests=${num_prompts}${entry_info}"
echo ""

# Helper: extract a metric line from a log file
# Usage: extract_metric <logfile> <grep_pattern>
extract_metric() {
    grep -E "$2" "$1" 2>/dev/null | tail -1
}

for rep in $(seq 1 ${NUM_REPS}); do
    echo "=========== Repetition ${rep} ==========="
    for variant in oriTopK heuristicTopK; do
        logfile=${log_prefix}_rep${rep}_${variant}.txt
        echo "  --- ${variant} ---"
        # Throughput & latency overview
        extract_metric "${logfile}" "^Request Throughput"            | sed 's/^/    /'
        extract_metric "${logfile}" "^Total Output Throughput"       | sed 's/^/    /'
        extract_metric "${logfile}" "^Total Token Throughput"        | sed 's/^/    /'
        extract_metric "${logfile}" "^Total Latency"                 | sed 's/^/    /'
        extract_metric "${logfile}" "^Average request latency"       | sed 's/^/    /'
        # Streaming metrics (TTFT, TPOT, generation speed)
        extract_metric "${logfile}" "^Average time-to-first-token"   | sed 's/^/    /'
        extract_metric "${logfile}" "^Average time-per-output-token" | sed 's/^/    /'
        extract_metric "${logfile}" "^Per User Output Speed"         | sed 's/^/    /'
        # TPOT percentiles
        grep -E "^\[TPOT\]" "${logfile}" 2>/dev/null | tail -7      | sed 's/^/    /'
        # TTFT percentiles
        grep -E "^\[TTFT\]" "${logfile}" 2>/dev/null | tail -7      | sed 's/^/    /'
        # Generation throughput per user
        grep -E "^\[GTPS\]" "${logfile}" 2>/dev/null | tail -7      | sed 's/^/    /'
        # Request latency percentiles
        grep -E "^\[Latency\]" "${logfile}" 2>/dev/null | tail -7   | sed 's/^/    /'
        # MTP speculative decoding stats
        grep -E "^\[DAR\]" "${logfile}" 2>/dev/null | tail -7       | sed 's/^/    /'
        grep -E "^\[AL\]" "${logfile}" 2>/dev/null | tail -7        | sed 's/^/    /'
        echo ""
    done
done

# --- Quick side-by-side comparison of key averages ---
echo "============================================"
echo "=== Quick A/B Comparison (averages) ==="
echo "============================================"
printf "%-6s  %12s %12s %12s %12s %12s %12s\n" "Rep" "TPOT(ori)" "TPOT(heur)" "TTFT(ori)" "TTFT(heur)" "DAR(ori)" "DAR(heur)"
for rep in $(seq 1 ${NUM_REPS}); do
    tpot_ori=$(grep -oP 'Average time-per-output-token \[TPOT\] \(ms\):\s+\K[\d.]+' ${log_prefix}_rep${rep}_oriTopK.txt 2>/dev/null | tail -1)
    tpot_heur=$(grep -oP 'Average time-per-output-token \[TPOT\] \(ms\):\s+\K[\d.]+' ${log_prefix}_rep${rep}_heuristicTopK.txt 2>/dev/null | tail -1)
    ttft_ori=$(grep -oP 'Average time-to-first-token \[TTFT\] \(ms\):\s+\K[\d.]+' ${log_prefix}_rep${rep}_oriTopK.txt 2>/dev/null | tail -1)
    ttft_heur=$(grep -oP 'Average time-to-first-token \[TTFT\] \(ms\):\s+\K[\d.]+' ${log_prefix}_rep${rep}_heuristicTopK.txt 2>/dev/null | tail -1)
    dar_ori=$(grep -E '^\[DAR\] AVERAGE' ${log_prefix}_rep${rep}_oriTopK.txt 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || true)
    dar_heur=$(grep -E '^\[DAR\] AVERAGE' ${log_prefix}_rep${rep}_heuristicTopK.txt 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || true)
    printf "%-6s  %12s %12s %12s %12s %12s %12s\n" "rep${rep}" "${tpot_ori:-N/A}" "${tpot_heur:-N/A}" "${ttft_ori:-N/A}" "${ttft_heur:-N/A}" "${dar_ori:-N/A}" "${dar_heur:-N/A}"
done
