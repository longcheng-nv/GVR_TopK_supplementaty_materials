#!/usr/bin/env python3
"""
Convert SWE-bench JSONL (system/user chat format) to trtllm-bench dataset format.

Reads the swe_bench_64k.jsonl file, applies the DeepSeek chat template to
tokenize each entry, and writes a trtllm-bench compatible JSONL with
pre-tokenized input_ids.

Usage:
    python prepare_swebench_dataset.py \
        --input /path/to/swe_bench_64k.jsonl \
        --tokenizer /path/to/model \
        --osl 512 \
        --output dataset_swebench.json
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench JSONL to trtllm-bench dataset format")
    parser.add_argument("--input", required=True,
                        help="Path to swe_bench_64k.jsonl")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to tokenizer / model directory")
    parser.add_argument("--osl", type=int, default=512,
                        help="Output sequence length per request")
    parser.add_argument("--output", default="dataset_swebench.json",
                        help="Output dataset file (JSONL)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)

    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} SWE-bench entries", file=sys.stderr)

    dataset = []
    for i, entry in enumerate(entries):
        messages = []
        if entry.get("system"):
            messages.append({"role": "system", "content": entry["system"]})
        messages.append({"role": "user", "content": entry["user"]})

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)

        dataset.append({
            "task_id": i,
            "input_ids": input_ids,
            "output_tokens": args.osl,
        })
        print(f"  Entry {i}: {len(input_ids)} input tokens, "
              f"osl={args.osl}", file=sys.stderr)

    # Write output
    with open(args.output, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    isls = [len(r["input_ids"]) for r in dataset]
    max_isl = max(isls)
    print(f"\nGenerated {len(dataset)} requests -> {args.output}",
          file=sys.stderr)
    print(f"  ISL range: [{min(isls)}, {max_isl}]", file=sys.stderr)
    print(f"  OSL: {args.osl}", file=sys.stderr)

    # Print max_isl to stdout so callers can capture it
    print(max_isl)


if __name__ == "__main__":
    main()
