# GVR Top-K Supplementary Materials

This branch collects the supplementary materials for the paper. The repository currently contains three main directories covering phase-level profiling, real long-context prompt inputs, and end-to-end A/B benchmarking on real SWE-bench prompts.

## Directory Overview


| Directory                                                 | Main content                                                             | Relationship to the others                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `gvr_phase_timing/`                                       | Tooling for breaking down GVR heuristic Top-K into four timing phases    | Focuses on kernel behavior analysis and explains where the Top-K speedup comes from                             |
| `longseqtasks/`                                           | Real SWE-bench long-context JSONL prompt datasets                        | Serves as the benchmark input source used directly by `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/` |
| `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/` | Heuristic Top-K vs original TopK A/B benchmark on real SWE-bench prompts | Accepts explicit SWE-bench JSONL paths, builds OSL-specific tokenized datasets, and runs E2E decode tests      |


## `gvr_phase_timing/`

This directory focuses on **four-phase timing analysis of the GVR heuristic Top-K kernel**. The goal is not to measure end-to-end throughput directly, but to decompose a single Top-K invocation into Guess / Verify / Collect / Refine phases and analyze the time distribution and dominant bottlenecks.

Key files:

- `gvr_phase_timing/README.md`: explains the motivation, workflow, expected outputs, and interpretation of phase-level timing results.
- `phase_timing_breakdown.patch`: patches the TensorRT-LLM heuristic Top-K kernel to write five `clock64()` timestamps into the tail of the scratch buffer.
- `phase_timing_breakdown_TUTORIAL.md`: explains how to apply and revert the patch, and how to read phase timestamps on the host side.
- `parse_phase_timing.py`: loads real decode logits, invokes the timed GVR Top-K kernel, parses the four phase durations, and writes a JSON summary.
- `run_phase_timing.sh`: one-click runner that wraps data path configuration, layer selection, and SM frequency setup.
- `ablation_preidx_experiment.py`: runs a preIdx quality ablation study, comparing no preIdx, random preIdx, previous-step Top-K, and related input signal quality variants.

Questions this directory helps answer:

- Which phase dominates the runtime of GVR Top-K?
- Is the Verify phase the main bottleneck across different layers or different prediction quality regimes?
- Can improvements in preIdx quality explain the observed heuristic Top-K speedup?

## `longseqtasks/`

This directory stores **real SWE-bench long-context prompt inputs** and serves as the raw data source for the downstream E2E benchmarks. It currently contains four JSONL files:

- `swe_bench_16k.jsonl`
- `swe_bench_32k.jsonl`
- `swe_bench_64k.jsonl`
- `swe_bench_100k.jsonl`

Common properties of these files:

- Each file contains 5 samples.
- Each sample uses chat-style JSONL in the form `{"system": "...", "user": "..."}`.
- In the current datasets, the `system` field is empty and the main content lives in the `user` field.
- The four files represent different input-length regimes, covering medium to very long contexts.

Role of this directory:

- It does not contain benchmark logic by itself; it is a reusable source of real prompt inputs.
- `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/` reads these JSONL files, tokenizes them, and converts them into the dataset format consumed by `trtllm-bench`.

## `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/`

This directory focuses on an **end-to-end A/B benchmark on real SWE-bench prompts**, comparing heuristic Top-K against the original radix-sort TopK under realistic decode workloads. Unlike `gvr_phase_timing/`, which concentrates on kernel-level breakdown, this directory emphasizes end-to-end metrics such as TTFT, TPOT, and DAR. The current script revision uses the TEP8/TRTLLM setup, caches OSL-specific tokenized datasets, and accepts explicit dataset paths or bare filenames resolved by the runner.

Key files:

- `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/README.md`: documents the benchmark motivation, parameters, input datasets, output log format, and usage examples.
- `run_perf_swebench.sh`: the main benchmark script; it accepts explicit raw JSONL paths (or bare filenames resolved by the runner), prepares OSL-specific tokenized datasets automatically, and alternates between original TopK and heuristic TopK runs.
- `prepare_swebench_dataset.py`: converts chat-format SWE-bench JSONL into the `input_ids` dataset format required by `trtllm-bench`.

Typical workflow:

1. Select a raw `swe_bench_*.jsonl` file and pass it via `--input` using an explicit path when possible.
2. Use `prepare_swebench_dataset.py` to tokenize it with the DeepSeek-V3.2-Exp chat template.
3. Let `run_perf_swebench.sh` compute `max_seq_len`, prepare the input dataset, and execute repeated A/B tests automatically.
4. Collect the benchmark logs and compare metrics such as TPOT, TTFT, and DAR.

Questions this directory helps answer:

- How much does heuristic Top-K improve E2E decode latency on real long-context coding prompts?
- Is the gain consistent across different input lengths (16K / 32K / 64K / 100K)?
- Do the kernel-level phase savings translate into measurable end-to-end benefits?

## How The Three Directories Fit Together

These three directories can be viewed as a pipeline from mechanism analysis to realistic workload validation:

1. `gvr_phase_timing/`: explains why heuristic Top-K is faster and which phases contribute the savings.
2. `longseqtasks/`: provides the real, reproducible long-context prompt inputs.
3. `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/`: performs the final end-to-end A/B validation on real prompts.

If you want a quick starting point:

- For the real benchmark workflow, start with `realistic_dataset_revised_swebench_E2E_decode_GVR_topK/README.md`
- For phase-level kernel analysis, start with `gvr_phase_timing/README.md`
- For the raw prompt inputs, start with `longseqtasks/`

