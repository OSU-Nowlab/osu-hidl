#!/usr/bin/env python3
"""
Simple SGLang Text Generation Example

This script demonstrates basic text generation using SGLang with a small model.
It showcases structured generation with control flow and multiple completions.

Usage:
    python simple_generation.py

Requirements:
    - sglang
    - transformers
    - torch
"""

import argparse
import sglang as sgl


@sgl.function
def simple_generation(s, prompt):
    """Simple text generation with SGLang."""
    s += prompt
    s += sgl.gen("output", max_tokens=100, temperature=0.8)


@sgl.function
def multi_turn_chat(s, question):
    """Multi-turn conversation example."""
    s += sgl.system("You are a helpful AI assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=150))


@sgl.function
def structured_generation(s, topic):
    """Structured generation with control flow."""
    s += f"Generate a story about {topic}.\n\n"

    # Title
    s += "Title: "
    s += sgl.gen("title", max_tokens=20, stop="\n")
    s += "\n\n"

    # Story
    s += "Story:\n"
    s += sgl.gen("story", max_tokens=200)

    # Moral
    s += "\n\nMoral of the story: "
    s += sgl.gen("moral", max_tokens=50, stop="\n")


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Simple Generation Example")
    parser.add_argument("--backend",
                        type=str,
                        default="runtime",
                        choices=["runtime", "openai"],
                        help="Backend to use (runtime or openai)")
    parser.add_argument("--model",
                        type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to use for generation")
    parser.add_argument("--host",
                        type=str,
                        default="127.0.0.1",
                        help="Server host (for OpenAI backend)")
    parser.add_argument("--port",
                        type=int,
                        default=30000,
                        help="Server port (for OpenAI backend)")
    args = parser.parse_args()

    # Set backend
    if args.backend == "runtime":
        sgl.set_default_backend(sgl.RuntimeEndpoint(args.model))
    else:
        sgl.set_default_backend(
            sgl.OpenAI(f"http://{args.host}:{args.port}/v1"))

    print("=" * 60)
    print("SGLang Simple Generation Examples")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print()

    # Example 1: Simple generation
    print("-" * 60)
    print("Example 1: Simple Text Generation")
    print("-" * 60)
    prompt = "Once upon a time in a distant land"
    state = simple_generation.run(prompt=prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {state['output']}")
    print()

    # Example 2: Multi-turn chat
    print("-" * 60)
    print("Example 2: Multi-turn Chat")
    print("-" * 60)
    question = "What is the capital of France?"
    state = multi_turn_chat.run(question=question)
    print(f"Question: {question}")
    print(f"Answer: {state['answer']}")
    print()

    # Example 3: Structured generation
    print("-" * 60)
    print("Example 3: Structured Generation")
    print("-" * 60)
    topic = "a robot discovering emotions"
    state = structured_generation.run(topic=topic)
    print(f"Topic: {topic}")
    print(f"Title: {state['title']}")
    print(f"Story: {state['story']}")
    print(f"Moral: {state['moral']}")
    print()

    print("=" * 60)
    print("All examples completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
