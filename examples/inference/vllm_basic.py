"""
Basic vLLM Inference Example

Demonstrates simple text generation using vLLM with a small model.
"""

from vllm import LLM, SamplingParams


def main():
    """Run basic inference with vLLM."""

    # Initialize model (will download on first run)
    print("Loading model...")
    llm = LLM(
        model="facebook/opt-125m",  # Small model for testing
        gpu_memory_utilization=0.5,
        max_num_seqs=8)

    # Define prompts
    prompts = [
        "Hello, my name is", "The capital of France is",
        "Artificial intelligence is", "The meaning of life is"
    ]

    # Configure generation parameters
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=50)

    # Generate text
    print("\nGenerating text...\n")
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
