#!/usr/bin/env python3
"""
Llama 2 Fine-tuning Benchmark
This script automates running multiple fine-tuning configurations with Axolotl
and tracks results with Weights & Biases.
"""

import os
import yaml
import json
import subprocess
import argparse
from datetime import datetime
import time
import shutil
from pathlib import Path


def create_config_variations():
    """Create different config variations for benchmarking."""
    
    # Base configuration
    base_config = {
        "base_model": "NousResearch/Llama-2-7b-hf",
        "model_type": "LlamaForCausalLM",
        "tokenizer_type": "LlamaTokenizer",
        "load_in_8bit": True,
        "load_in_4bit": False,
        "strict": False,
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca"
            }
        ],
        "val_set_size": 0.05,
        "sequence_len": 4096,
        "sample_packing": True,
        "eval_sample_packing": False,
        "pad_to_sequence_len": True,
        "adapter": "lora",
        "lora_r": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_linear": True,
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 2,
        "num_epochs": 4,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.0002,
        "train_on_inputs": False,
        "group_by_length": False,
        "bf16": "auto",
        "gradient_checkpointing": True,
        "warmup_steps": 10,
        "evals_per_epoch": 4,
        "eval_max_new_tokens": 128,
        "saves_per_epoch": 1,
        "weight_decay": 0.0,
        "flash_attention": True,
        "logging_steps": 1,
    }
    
    # Create refined list of variations as requested
    variations = [
        # Baseline - no changes
        {"name": "baseline", "config": base_config.copy(), "output_dir": "./outputs/benchmark/baseline"},
        
        # Batch size variations
        {"name": "batch_size_1", 
         "config": {**base_config.copy(), "micro_batch_size": 1}, 
         "output_dir": "./outputs/benchmark/batch_size_1"},
        {"name": "batch_size_4", 
         "config": {**base_config.copy(), "micro_batch_size": 4}, 
         "output_dir": "./outputs/benchmark/batch_size_4"},
        
        # Sequence length variations
        {"name": "seq_length_2048", 
         "config": {**base_config.copy(), "sequence_len": 2048}, 
         "output_dir": "./outputs/benchmark/seq_length_2048"},
        {"name": "seq_length_8192", 
         "config": {**base_config.copy(), "sequence_len": 8192}, 
         "output_dir": "./outputs/benchmark/seq_length_8192"},
        
        # LoRA rank variations
        {"name": "lora_r_16", 
         "config": {**base_config.copy(), "lora_r": 16, "lora_alpha": 8}, 
         "output_dir": "./outputs/benchmark/lora_r_16"},
        {"name": "lora_r_64", 
         "config": {**base_config.copy(), "lora_r": 64, "lora_alpha": 32}, 
         "output_dir": "./outputs/benchmark/lora_r_64"},
        
        # 4-bit quantization
        {"name": "quantization_4bit", 
         "config": {**base_config.copy(), "load_in_8bit": False, "load_in_4bit": True}, 
         "output_dir": "./outputs/benchmark/quantization_4bit"},
        
        # Larger model variation
        {"name": "larger_model", 
         "config": {
             **base_config.copy(), 
             "base_model": "NousResearch/Llama-2-13b-hf"  # Using 13B parameter model instead of 7B
         }, 
         "output_dir": "./outputs/benchmark/larger_model"},
        
        # Alternative fine-tuning strategy: QLoRA
        {"name": "qlora", 
         "config": {
             **base_config.copy(),
             "adapter": "qlora",
             "load_in_8bit": False,
             "load_in_4bit": True,
             "quantization_config": {
                 "bnb_4bit_compute_dtype": "float16",
                 "bnb_4bit_quant_type": "nf4",
                 "bnb_4bit_use_double_quant": True
             }
         }, 
         "output_dir": "./outputs/benchmark/qlora"},
    ]
    
    # Add W&B configuration to each variation
    for variation in variations:
        # Configure W&B for experiment tracking
        variation["config"]["wandb_project"] = "llama2-finetuning-benchmark"
        variation["config"]["wandb_entity"] = "sami-agourram-college-of-computing"
        variation["config"]["wandb_name"] = variation["name"]
        variation["config"]["wandb_log_model"] = "checkpoint"
    
    return variations


def save_config(config, output_path):
    """Save a configuration to a YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


def run_experiment(config_path):
    """Run a single experiment using Axolotl with the specified config."""
    cmd = f"accelerate launch -m axolotl.cli.train {config_path}"
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Get final return code and any remaining output
    process.communicate()
    
    if process.returncode != 0:
        print(f"Experiment failed with return code {process.returncode}")
        return False
    
    return True


def extract_results(output_dir):
    """Extract key results from the trainer_state.json file."""
    state_file = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        return None
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Extract final metrics
    best_eval_loss = None
    for log in state.get('log_history', []):
        if 'eval_loss' in log:
            if best_eval_loss is None or log['eval_loss'] < best_eval_loss:
                best_eval_loss = log['eval_loss']
    
    # Get training time
    if state.get('start_time') and state.get('end_time'):
        training_time = state['end_time'] - state['start_time']
    else:
        training_time = None
    
    return {
        'best_eval_loss': best_eval_loss,
        'training_time': training_time,
        'total_steps': state.get('global_step'),
    }


def compile_results(variations):
    """Compile results from all experiments into a single table."""
    results = []
    
    for variation in variations:
        output_dir = variation["output_dir"]
        result = extract_results(output_dir)
        
        if result:
            results.append({
                'name': variation['name'],
                **result
            })
    
    return results


def save_results_table(results, output_path):
    """Save the results table to a markdown and CSV file."""
    if not results:
        return
    
    # Create markdown table
    md_lines = ["# Llama 2 Fine-tuning Benchmark Results", ""]
    md_lines.append("| Experiment | Best Eval Loss | Training Time (s) | Total Steps |")
    md_lines.append("|------------|----------------|-------------------|-------------|")
    
    # Create CSV data
    csv_lines = ["Experiment,Best Eval Loss,Training Time (s),Total Steps"]
    
    for result in results:
        md_line = f"| {result['name']} | {result.get('best_eval_loss', 'N/A'):.4f} | {result.get('training_time', 'N/A'):.1f} | {result.get('total_steps', 'N/A')} |"
        md_lines.append(md_line)
        
        csv_line = f"{result['name']},{result.get('best_eval_loss', 'N/A'):.4f},{result.get('training_time', 'N/A'):.1f},{result.get('total_steps', 'N/A')}"
        csv_lines.append(csv_line)
    
    # Save markdown
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path + '.md', 'w') as f:
        f.write('\n'.join(md_lines))
    
    # Save CSV
    with open(output_path + '.csv', 'w') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"Results saved to {output_path}.md and {output_path}.csv")


def main():
    parser = argparse.ArgumentParser(description="Run Llama 2 fine-tuning benchmark")
    parser.add_argument("--configs_dir", default="./configs", help="Directory to save generated config files")
    parser.add_argument("--results_dir", default="./results", help="Directory to save results")
    parser.add_argument("--selected", nargs="*", help="Only run selected configurations (by name)")
    parser.add_argument("--skip", nargs="*", help="Skip specific configurations (by name)")
    
    args = parser.parse_args()
    
    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs_dir = os.path.join(args.configs_dir, timestamp)
    results_dir = os.path.join(args.results_dir, timestamp)
    
    # Get variations
    all_variations = create_config_variations()
    
    # Filter variations based on arguments
    if args.selected:
        variations = [v for v in all_variations if v["name"] in args.selected]
        print(f"Running only selected configurations: {[v['name'] for v in variations]}")
    elif args.skip:
        variations = [v for v in all_variations if v["name"] not in args.skip]
        print(f"Skipping configurations: {args.skip}")
    else:
        variations = all_variations
    
    # Save all configurations
    configs = {}
    for variation in variations:
        config_path = os.path.join(configs_dir, f"{variation['name']}.yml")
        save_config(variation["config"], config_path)
        configs[variation["name"]] = config_path
        print(f"Created config: {config_path}")
    
    # Run each experiment
    print(f"\nStarting benchmark with {len(variations)} configurations")
    for i, variation in enumerate(variations):
        print(f"\n[{i+1}/{len(variations)}] Running experiment: {variation['name']}")
        print(f"Output directory: {variation['output_dir']}")
        
        # Make sure output directory exists
        os.makedirs(variation["output_dir"], exist_ok=True)
        
        # Run the experiment
        config_path = configs[variation["name"]]
        start_time = time.time()
        success = run_experiment(config_path)
        elapsed = time.time() - start_time
        
        print(f"Experiment '{variation['name']}' {'completed' if success else 'failed'} in {elapsed:.1f} seconds")
    
    # Compile and save results
    print("\nCompiling results...")
    results = compile_results(variations)
    
    results_path = os.path.join(results_dir, "benchmark_results")
    save_results_table(results, results_path)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()