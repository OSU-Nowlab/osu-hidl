"""
vLLM Multi-GPU Inference Example

Demonstrates tensor parallelism across multiple GPUs for efficient inference.

Requirements:
- Multiple GPUs (2+)
- Large model (7B+ recommended to see benefits)
"""

from vllm import LLM, SamplingParams
import torch


def main():
    """Run multi-GPU inference with vLLM."""

    # Check available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    if gpu_count < 2:
        print(
            "Warning: Only 1 GPU detected. Multi-GPU features won't be demonstrated."
        )
        print("Run on a node with multiple GPUs for full demonstration.")
        tensor_parallel_size = 1
    else:
        tensor_parallel_size = min(gpu_count, 2)  # Use up to 2 GPUs
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")

    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\nLoading model with tensor parallelism...")

    # Initialize model with multi-GPU support
    llm = LLM(
        model="facebook/opt-1.3b",  # Medium-sized model
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_num_seqs=16  # Higher batch size for throughput
    )

    # Create a larger batch to demonstrate throughput
    prompts = [f"Write a haiku about the number {i}." for i in range(16)]

    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=50)

    # Generate (vLLM will distribute work across GPUs)
    print(
        f"\nGenerating {len(prompts)} responses using {tensor_parallel_size} GPU(s)..."
    )
    outputs = llm.generate(prompts, sampling_params)

    # Display first few results
    print("\nSample outputs:\n")
    for i, output in enumerate(outputs[:3], 1):
        print(f"Prompt {i}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 60)

    print(f"\n... and {len(outputs) - 3} more generated successfully.")
    print(f"\nTotal responses: {len(outputs)}")
    print("Multi-GPU inference complete!")


if __name__ == "__main__":
    main()
