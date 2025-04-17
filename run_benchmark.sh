#!/bin/bash
# Run the Llama 2 fine-tuning benchmark

# Check if the script has been made executable
if [[ ! -x "./llama_benchmark.py" ]]; then
    chmod +x ./llama_benchmark.py
    echo "Made llama_benchmark.py executable"
fi

# Login to Weights & Biases (first time only)
# wandb login

# Run the full benchmark
echo "Starting full benchmark..."
./llama_benchmark.py

# Or run a specific subset of configurations
# ./llama_benchmark.py --selected baseline batch_size_4 lr_high

# Or skip certain configurations
# ./llama_benchmark.py --skip quantization_4bit seq_length_8192