# Recipes

Reproducible end-to-end workflows for common ML tasks with the HPC-AI stack.

## Overview

While `examples/` contains simple code snippets demonstrating individual features, `recipes/` provides **complete, production-ready workflows** including scripts, configs, and expected results. Each recipe is designed to be copied and customized for your use case.

## Available Recipes

| Recipe | Description | Requirements |
|--------|-------------|--------------|
| [llama2-lora-finetune](llama2-lora-finetune/) | LoRA fine-tuning with DeepSpeed ZeRO | 1+ GPU, 16GB+ VRAM |
| [serving-comparison](serving-comparison/) | vLLM vs SGLang throughput comparison | 1+ GPU, 16GB+ VRAM |
| [comm-validation](comm-validation/) | MPI communication validation | 2+ GPUs or nodes |
## Quick Start

```bash
# Navigate to a recipe
cd recipe/llama2-lora-finetune

# Read the README for requirements
cat README.md

# Run the recipe
bash run.sh
```

## Recipe Structure

Each recipe follows a standard structure:

```
recipe-name/
├── README.md           # Documentation and expected results
├── run.sh              # Main entry point
├── *.py                # Python scripts
└── configs/            # Configuration files (if needed)
```

## Creating Your Own Recipe

1. Copy an existing recipe as a starting point
2. Modify the configuration for your model/data
3. Update the README with your expected results
4. Test on a small scale before scaling up

---

**Maintained by**: NOWLAB Team, The Ohio State University
