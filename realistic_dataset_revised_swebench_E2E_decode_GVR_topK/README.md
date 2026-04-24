# SWE-bench Heuristic TopK A/B Benchmark

A/B benchmark measuring the E2E decode latency impact of **heuristic TopK vs original radix-sort TopK** for DeepSeek-V3.2-Exp sparse attention (DSA) on 8x B200 GPUs, using real SWE-bench coding prompts.

The current script revision matches the TEP8 setup used in the latest experiments: `enable_attention_dp: false`, `moe_config.backend: TRTLLM`, and speculative decoding enabled only when `--MTP > 0`.

## Motivation

The TopK indexer in sparse attention is a significant fraction of E2E decode time at batch size 1. Heuristic TopK aims to reduce this cost. This benchmark uses real SWE-bench prompts (not random tokens) to produce realistic token distributions and stable MTP acceptance rates, giving representative TPOT measurements for the A/B comparison.

## Prerequisites

- 8x NVIDIA B200 GPUs
- TensorRT-LLM with `trtllm-bench` on PATH (typically `~/.local/bin`)
- DeepSeek-V3.2-Exp FP4 model available locally, provided via `MODEL_PATH=/path/to/DeepSeek-V3.2-Exp-FP4-v2/`
- Python 3 with `transformers` installed (for tokenization)
- `sudo` access for GPU power management setup

## Files

| File | Description |
|------|-------------|
| `run_perf_swebench.sh` | Main A/B benchmark script |
| `prepare_swebench_dataset.py` | Converts SWE-bench JSONL to trtllm-bench format |
| `--input <path>` | Raw SWE-bench JSONL input; can be an explicit path or a bare filename |
| `dataset_swebench_*_osl*.json` | Generated tokenized datasets, keyed by both ISL tag and OSL |
| `dataset_swebench_*_osl*.json.max_isl` | Cached max-ISL sidecar written by the runner |

## Quick Start

```bash
# Run with an explicit dataset path
bash run_perf_swebench.sh --input /path/to/swe_bench_64k.jsonl
```

## Usage

```bash
run_perf_swebench.sh [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input <file>` | SWE-bench JSONL file; checks the given path, then `./`, then the bundled `../longseqtasks/` directory for bare names | `swe_bench_64k.jsonl` |
| `--entry <N>` | Run only entry N (0-indexed) from the dataset | all entries |
| `--num-requests <N>` | Number of requests to benchmark | all entries, or 10 for single entry |
| `--dataset_osl <N>` | Output sequence length per request | `2025` |
| `--MTP <N>` | Number of MTP speculative decoding layers | `1` |

Positional arguments are also supported for backward compatibility: `run_perf_swebench.sh [ENTRY_ID] [NUM_REQUESTS]`.

For reproducibility, passing an explicit `--input /full/path/to/swe_bench_*.jsonl` is recommended. If a bare filename is given, the script first checks the current working directory and then falls back to the bundled `../longseqtasks/` directory.

### Examples

```bash
# Recommended: explicit dataset path
bash run_perf_swebench.sh --input /path/to/swe_bench_64k.jsonl

# Or rely on bare filename resolution if the file exists in the default tasks dir
bash run_perf_swebench.sh --input swe_bench_64k.jsonl

# Use the 100K token dataset
bash run_perf_swebench.sh --input /path/to/swe_bench_100k.jsonl

# Single prompt: entry #2 from the 64K dataset, 10 requests
bash run_perf_swebench.sh --input /path/to/swe_bench_64k.jsonl --entry 2

# Single prompt with custom request count
bash run_perf_swebench.sh --input /path/to/swe_bench_64k.jsonl --entry 2 --num-requests 20

# 100K dataset, specific entry
bash run_perf_swebench.sh --input /path/to/swe_bench_100k.jsonl --entry 1

# Custom OSL and MTP settings
bash run_perf_swebench.sh --input /path/to/swe_bench_64k.jsonl --dataset_osl 512 --MTP 3
```

## Available Input Datasets

This README tracks the two long-context datasets used in the revised supplementary benchmark: `swe_bench_64k.jsonl` and `swe_bench_100k.jsonl`. Copies may live in `../longseqtasks/` or `../../deepseek-v3.2-logging/tasks/`; pass whichever path exists in your environment via `--input`. Each file contains 5 real SWE-bench coding prompts in `{"system": "", "user": "..."}` JSONL format.

### `swe_bench_64k.jsonl`

| Entry | ISL (tokens) |
|-------|-------------|
| 0 | 57,615 |
| 1 | 68,656 |
| 2 | 52,244 |
| 3 | 62,647 |
| 4 | 64,415 |

ISL range: **52,244 -- 68,656**. Widest ISL spread across entries; good for testing TopK sensitivity to context length variation.

### `swe_bench_100k.jsonl`

| Entry | ISL (tokens) |
|-------|-------------|
| 0 | 100,200 |
| 1 | 101,222 |
| 2 | 101,781 |
| 3 | 103,740 |
| 4 | 104,749 |

ISL range: **100,200 -- 104,749**. Longest prompts; maximizes the number of sparse attention blocks to index, where heuristic TopK should show the largest advantage.

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-V3.2-Exp FP4 |
| Batch size | 1 (min-latency) |
| OSL | 2025 (configurable via `--dataset_osl`) |
| MTP | 1 (configurable via `--MTP`) |
| Parallelism | EP=8, TEP-style config with attention DP disabled |
| MoE backend | TRTLLM |
| KV cache | FP8, 80% GPU memory, 64 tokens/block |
| CUDA graphs | Enabled with padding |
| `max_seq_len` | Auto-computed: `ceil(1.25 * max_ISL / 1024) * 1024` |
| Repetitions | 3 interleaved A/B pairs |

## How It Works

1. **GPU setup** -- Sets persistence mode and vboost 4.
2. **Input resolution** -- `--input` may be an explicit path, a local file, or a bare filename that resolves under `../../deepseek-v3.2-logging/tasks/`.
3. **Dataset preparation** -- On first run, `prepare_swebench_dataset.py` tokenizes the selected SWE-bench JSONL and writes an OSL-specific cache file named `dataset_swebench_<isl>_osl<osl>.json`. The runner writes to a temporary file first, atomically renames it into place, and stores `max_isl` in a `.max_isl` sidecar for reuse.
4. **Entry selection** -- If `--entry` is specified, extracts that single prompt into a separate file.
5. **A/B loop** (3 repetitions, interleaved) -- Each repetition runs:
   - **Run A**: `enable_heuristic_topk: false` (original radix-sort TopK)
   - **Run B**: `enable_heuristic_topk: true` (heuristic TopK)
6. **Results summary** -- Prints detailed metrics per variant per repetition, plus a quick side-by-side comparison table.

## Output

### Log Files

Logs are written to:
```
./tmp/ds_swebench_BS1_ISL{tag}_OSL{osl}_fp4_heuristic_topk_perf_CUDAGraph[_entry{N}]/
    run_swebench[_entry{N}]_DEP8_1_8192_FP4_B200_MTP{M}_{timestamp}_rep{R}_{oriTopK|heuristicTopK}.txt
```

### Metrics Reported

**Per-variant detailed output:**
- Request throughput (req/s), output throughput (tok/s), total token throughput (tok/s)
- Total latency, average request latency
- TTFT (time-to-first-token), TPOT (time-per-output-token)
- Per-user output speed (tps/user)
- Percentile breakdowns (min/max/avg/p50/p90/p95/p99) for: TPOT, TTFT, GTPS, request latency
- MTP speculative decoding: DAR (draft acceptance rate), AL (acceptance length)

**Quick A/B comparison table:**
```
Rep     TPOT(ori)  TPOT(heur)   TTFT(ori)  TTFT(heur)    DAR(ori)   DAR(heur)
rep1       8.35        7.92     13936.77    13940.12        0.57       0.58
rep2       ...         ...         ...         ...          ...        ...
rep3       ...         ...         ...         ...          ...        ...
```

## Dataset Preparation Script

`prepare_swebench_dataset.py` can also be used standalone:

```bash
python3 prepare_swebench_dataset.py \
    --input /path/to/swe_bench_64k.jsonl \
    --tokenizer PATH/TO/DeepSeek-V3.2-Exp-FP4-v2/ \
    --osl 512 \
    --output dataset_swebench.json
```

Input: JSONL with `{"system": "...", "user": "..."}` per line.
Output: JSONL with `{"task_id": N, "input_ids": [...], "output_tokens": 512}` per line.
Prints `max_isl` to stdout for programmatic use.
