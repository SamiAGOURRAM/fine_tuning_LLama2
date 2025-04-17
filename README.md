# Llama 2 Fine-tuning Benchmark with Axolotl

This repository contains a benchmarking system for systematically evaluating different hyperparameter configurations when fine-tuning Llama 2 with Axolotl. The system is designed to automatically run multiple configurations, track results with Weights & Biases, and analyze performance across different hyperparameter settings.

## Features

- 🔄 **Automated benchmarking**: Run multiple fine-tuning configurations sequentially
- 📊 **Experiment tracking**: Integration with Weights & Biases for detailed experiment logging
- 📈 **Visualization**: Generate charts and reports comparing different configurations
- 🛠️ **Customizable**: Easily add or modify configurations to test

## Getting Started

### Prerequisites

- Python 3.8+
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) installed
- A GPU with at least 24GB VRAM (recommended)
- [Weights & Biases](https://wandb.ai) account (free tier available)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llama2-finetuning-benchmark.git
   cd llama2-finetuning-benchmark
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Login to Weights & Biases:
   ```
   wandb login
   ```

### Usage

1. Make scripts executable:
   ```
   chmod +x llama_benchmark.py analyze_results.py run_benchmark.sh
   ```

2. Run the benchmark:
   ```
   ./run_benchmark.sh
   ```

   Or to run specific configurations:
   ```
   ./llama_benchmark.py --selected baseline batch_size_4 seq_length_8192
   ```

3. Analyze results locally:
   ```
   ./analyze_results.py
   ```

## Configurations

The benchmark includes the following configurations:

- **Baseline**: The default Llama-2-7b configuration with LoRA
- **Batch Size**: Variations in micro_batch_size (1, 4)
- **Sequence Length**: Variations in sequence_len (2048, 8192)
- **LoRA Rank**: Variations in lora_r and lora_alpha (16/8, 64/32)
- **Quantization**: 4-bit quantization vs 8-bit baseline
- **Model Size**: Llama-2-13b vs Llama-2-7b baseline
- **Fine-tuning Strategy**: QLoRA vs standard LoRA

## Weights & Biases Visualizations

All experiments are automatically tracked in Weights & Biases. You can access your W&B dashboard directly for in-depth analysis and visualization:

1. After running the benchmark, go to [https://wandb.ai](https://wandb.ai) and log in to your account.

2. Navigate to your project (default: `llama2-finetuning-benchmark`).

3. Use W&B's visualization tools to:
   - Compare training and evaluation loss across runs
   - Track GPU memory usage and training time
   - Create custom visualizations and reports
   - Share results with collaborators

### Creating W&B Reports

W&B Reports provide a powerful way to document and share your experiments:

1. In your W&B dashboard, click on "Create Report"
2. Select the runs you want to include in your analysis
3. Add panels like line charts, bar charts, parameter importance, etc.
4. Save and share your report with a simple link

You can create visualizations like:
- Loss curves comparing all configurations
- Bar charts of final validation loss
- Training time comparisons
- GPU memory usage across configurations
- Parameter importance analysis

The W&B interface offers more flexibility than our local visualization script, particularly for interactive exploration and creating publication-quality charts.