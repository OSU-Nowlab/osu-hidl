#!/usr/bin/env python3
"""
vLLM Throughput Benchmark

Measures inference throughput and latency for vLLM.

Usage:
    python benchmark_vllm.py
    python benchmark_vllm.py --model facebook/opt-6.7b --num_prompts 100
"""

import argparse
import time
import json

# Optional import with graceful fallback
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='vLLM Throughput Benchmark')
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
    parser.add_argument('--warmup',
                        type=int,
                        default=5,
                        help='Number of warmup iterations')
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='Output file for results (JSON)')
    return parser.parse_args()


def create_prompts(num_prompts):
    """Create diverse test prompts."""
    base_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy?",
        "Describe the process of making coffee.",
        "How does the internet work?",
        "What is the meaning of life?",
        "Explain quantum computing to a child.",
        "Write a poem about the ocean.",
        "What are the key principles of good software design?",
        "Describe the solar system.",
    ]

    prompts = []
    for i in range(num_prompts):
        prompts.append(base_prompts[i % len(base_prompts)])

    return prompts


def run_benchmark(args):
    """Run vLLM benchmark."""
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("")

    # Initialize vLLM
    print("Loading model...")
    llm = LLM(model=args.model,
              tensor_parallel_size=args.tensor_parallel_size,
              dtype="float16",
              gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=args.max_tokens)

    # Create prompts
    prompts = create_prompts(args.num_prompts)
    warmup_prompts = prompts[:args.warmup]

    # Warmup
    print(f"Warming up ({args.warmup} prompts)...")
    _ = llm.generate(warmup_prompts, sampling_params)

    # Benchmark
    print(f"Running benchmark ({args.num_prompts} prompts)...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    # Calculate metrics
    elapsed = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    avg_latency = (elapsed / args.num_prompts) * 1000  # ms

    results = {
        "framework": "vLLM",
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "throughput_tokens_per_sec": throughput,
        "avg_latency_ms": avg_latency
    }

    # Print results
    print("")
    print("Results:")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Throughput: {throughput:,.1f} tokens/sec")
    print(f"  Avg latency: {avg_latency:.1f} ms/request")

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


def main():
    """Main entry point."""
    if not HAS_VLLM:
        print("ERROR: vLLM is not installed.")
        print("Install with: pip install vllm")
        return None

    args = get_args()

    print("=" * 60)
    print("vLLM Throughput Benchmark")
    print("=" * 60)

    results = run_benchmark(args)

    print("")
    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
