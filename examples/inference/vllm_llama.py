"""
vLLM Llama-2 Inference Example

Demonstrates text generation using Llama-2 models with vLLM.

Requirements:
- GPU with sufficient memory (7B model requires ~14GB)
- HuggingFace token for gated models (if using official Llama-2)
"""

from vllm import LLM, SamplingParams
import os


def main():
    """Run inference with Llama-2."""

    # Choose model size (7B, 13B, or 70B)
    # Note: You may need HuggingFace authentication for official Meta models
    model_name = "meta-llama/Llama-2-7b-hf"

    # Alternative: Use open Llama-2 compatible models
    # model_name = "NousResearch/Llama-2-7b-hf"

    print(f"Loading {model_name}...")
    print("Note: First run will download the model (~13GB for 7B)")

    # Initialize model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Use 1 GPU (set to 2 for larger models)
        dtype="float16",
        gpu_memory_utilization=0.9)

    # Define prompts
    prompts = [
        "Write a short story about a robot learning to paint:",
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate Fibonacci numbers:"
    ]

    # Configure generation
    sampling_params = SamplingParams(temperature=0.7,
                                     top_p=0.9,
                                     max_tokens=200,
                                     presence_penalty=0.1)

    # Generate
    print("\nGenerating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    for i, output in enumerate(outputs, 1):
        print(f"=== Example {i} ===")
        print(f"Prompt: {output.prompt}\n")
        print(f"Response: {output.outputs[0].text}\n")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    # Check for HuggingFace token if needed
    if "meta-llama" in "meta-llama/Llama-2-7b-hf":
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "Warning: HF_TOKEN not set. You may need it for official Llama-2 models."
            )
            print("Set with: export HF_TOKEN=your_token_here")
            print(
                "Or use alternative models like NousResearch/Llama-2-7b-hf\n")

    main()
