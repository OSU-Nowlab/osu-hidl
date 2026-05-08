#!/usr/bin/env python3
"""
Llama-2 LoRA Fine-tuning with DeepSpeed

Fine-tune Llama-2 models using LoRA (Low-Rank Adaptation) with DeepSpeed
ZeRO optimization for memory-efficient training.

Usage:
    # Quick test with TinyLlama
    deepspeed finetune.py --deepspeed_config=ds_config.json

    # With Llama-2-7B
    deepspeed finetune.py --model meta-llama/Llama-2-7b-hf --deepspeed_config=ds_config.json

Requirements:
    - transformers
    - peft
    - deepspeed
    - datasets
"""

import argparse
import os
import torch
from torch.utils.data import Dataset
import deepspeed

# Optional imports with graceful fallback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing when real data is not available."""

    def __init__(self, tokenizer, size=1000, seq_length=128):
        self.size = size
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        # Create synthetic prompts
        self.prompts = [
            "Explain machine learning in simple terms.",
            "What is the capital of France?",
            "Write a poem about the ocean.",
            "How do computers work?",
            "Describe the process of photosynthesis.",
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]

        # Tokenize
        encoded = self.tokenizer(prompt,
                                 max_length=self.seq_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Llama-2 LoRA Fine-tuning')

    # Model arguments
    parser.add_argument('--model',
                        type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='Model name or path')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./output',
                        help='Output directory for LoRA adapter')

    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha',
                        type=int,
                        default=16,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout',
                        type=float,
                        default=0.05,
                        help='LoRA dropout')

    # Training arguments
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch size per GPU')
    parser.add_argument('--max_steps',
                        type=int,
                        default=100,
                        help='Maximum training steps (for testing)')
    parser.add_argument('--seq_length',
                        type=int,
                        default=128,
                        help='Maximum sequence length')

    # DeepSpeed arguments
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='Local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def print_rank0(msg):
    """Print only from rank 0."""
    if deepspeed.comm.get_rank() == 0:
        print(msg)


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    """Main training function."""
    args = get_args()

    # Check dependencies
    if not HAS_TRANSFORMERS:
        print("ERROR: This recipe requires transformers and peft packages.")
        print("Install with: pip install transformers peft datasets")
        return

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    print_rank0("=" * 60)
    print_rank0("Llama-2 LoRA Fine-tuning with DeepSpeed")
    print_rank0("=" * 60)
    print_rank0(f"Model: {args.model}")
    print_rank0(f"LoRA rank: {args.lora_r}")
    print_rank0(f"DeepSpeed config: {args.deepspeed_config}")
    print_rank0("")

    # Load tokenizer
    print_rank0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print_rank0("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)

    # Configure LoRA
    print_rank0("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none")

    model = get_peft_model(model, lora_config)

    # Print parameter counts
    trainable, total = count_parameters(model)
    print_rank0(
        f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )
    print_rank0("")

    # Create dataset
    print_rank0("Creating dataset...")
    dataset = SyntheticDataset(tokenizer,
                               size=1000,
                               seq_length=args.seq_length)

    # Initialize DeepSpeed engine
    print_rank0("Initializing DeepSpeed engine...")
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        model_parameters=model.parameters())

    # Training loop
    print_rank0(
        f"\nTraining for {args.epochs} epochs (max {args.max_steps} steps)...")
    print_rank0("")

    global_step = 0
    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            # Move to device
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(
                model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            # Forward pass
            outputs = model_engine(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
            loss = outputs.loss

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print_rank0(
                f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

        if global_step >= args.max_steps:
            print_rank0(
                f"Reached max_steps ({args.max_steps}), stopping early.")
            break

    # Save LoRA adapter
    print_rank0("")
    if deepspeed.comm.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        adapter_path = os.path.join(args.output_dir, "lora_adapter")
        model_engine.module.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print_rank0(f"Adapter saved to: {adapter_path}")

    print_rank0("")
    print_rank0("=" * 60)
    print_rank0("Training complete!")
    print_rank0("=" * 60)


if __name__ == "__main__":
    main()
