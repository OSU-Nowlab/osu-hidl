#!/usr/bin/env python3
"""
vLLM vs SGLang Comparison

Runs both frameworks and compares throughput and latency.

Usage:
    python compare.py
    python compare.py --model facebook/opt-1.3b --num_prompts 50
"""

import argparse
import subprocess
import json
import os
import tempfile


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='vLLM vs SGLang Comparison')
    parser.add_argument('--model',
                        type=str,
                        default='facebook/opt-1.3b',
                        help='Model name or path')
    parser.add_argument('--num_prompts',
                        type=int,
                        default=50,
                        help='Number of prompts to process')
    parser.add_argument('--max_tokens',
                        type=int,
                        default=128,
                        help='Maximum tokens to generate')
    parser.add_argument('--vllm_only',
                        action='store_true',
                        help='Only run vLLM benchmark')
    parser.add_argument('--sglang_only',
                        action='store_true',
                        help='Only run SGLang benchmark')
    return parser.parse_args()


def run_benchmark(script, args, output_file):
    """Run a benchmark script and capture results."""
    cmd = [
        "python", script, "--model", args.model, "--num_prompts",
        str(args.num_prompts), "--max_tokens",
        str(args.max_tokens), "--output", output_file
    ]

    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                timeout=600)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if os.path.exists(output_file):
            with open(output_file) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
    except Exception as e:
        print(f"Error running benchmark: {e}")

    return None


def print_comparison(vllm_results, sglang_results):
    """Print comparison table."""
    print("")
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print("")
    print(f"{'Metric':<20} {'vLLM':>12} {'SGLang':>12} {'Winner':>10}")
    print("-" * 60)

    # Throughput comparison
    vllm_tp = vllm_results.get("throughput_tokens_per_sec", 0)
    sglang_tp = sglang_results.get("throughput_tokens_per_sec", 0)
    winner = "vLLM" if vllm_tp > sglang_tp else "SGLang"
    print(
        f"{'Throughput (t/s)':<20} {vllm_tp:>12,.1f} {sglang_tp:>12,.1f} {winner:>10}"
    )

    # Latency comparison
    vllm_lat = vllm_results.get("avg_latency_ms", 0)
    sglang_lat = sglang_results.get("avg_latency_ms", 0)
    winner = "vLLM" if vllm_lat < sglang_lat else "SGLang"
    print(
        f"{'Avg Latency (ms)':<20} {vllm_lat:>12,.1f} {sglang_lat:>12,.1f} {winner:>10}"
    )

    # Total time
    vllm_time = vllm_results.get("elapsed_seconds", 0)
    sglang_time = sglang_results.get("elapsed_seconds", 0)
    winner = "vLLM" if vllm_time < sglang_time else "SGLang"
    print(
        f"{'Total Time (s)':<20} {vllm_time:>12,.2f} {sglang_time:>12,.2f} {winner:>10}"
    )

    print("-" * 60)
    print("")


def main():
    """Main entry point."""
    args = get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("LLM Serving Framework Comparison")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print("")

    vllm_results = None
    sglang_results = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run vLLM benchmark
        if not args.sglang_only:
            print("Running vLLM benchmark...")
            print("-" * 40)
            vllm_output = os.path.join(tmpdir, "vllm_results.json")
            vllm_results = run_benchmark(
                os.path.join(script_dir, "benchmark_vllm.py"), args,
                vllm_output)
            print("")

        # Run SGLang benchmark
        if not args.vllm_only:
            print("Running SGLang benchmark...")
            print("-" * 40)
            sglang_output = os.path.join(tmpdir, "sglang_results.json")
            sglang_results = run_benchmark(
                os.path.join(script_dir, "benchmark_sglang.py"), args,
                sglang_output)
            print("")

    # Print comparison if both ran
    if vllm_results and sglang_results:
        print_comparison(vllm_results, sglang_results)
    elif vllm_results:
        print("\nOnly vLLM results available.")
    elif sglang_results:
        print("\nOnly SGLang results available.")
    else:
        print("\nNo results to compare. Check that frameworks are installed.")

    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
