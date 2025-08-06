#!/usr/bin/env python3
"""
Full GRPO training script with async generation using MLX RL inference server.

Usage:
    python train_async_grpo.py --server-url http://localhost:8000 --model-path /path/to/model

Prerequisites:
    1. Start the MLX inference server with your base model:
       python -m mlx_lm.inference_server --model <model_path>
    2. Ensure the server is running and accessible
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load, save_model
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.trainer import TrainingArgs
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters



# Import GRPO training components
from mlx_lm_lora.trainer.grpo_trainer import (
    GRPOTrainingArgs, 
    train_grpo,
    iterate_grpo_batches,
)
from mlx_lm_lora.trainer.grpo_reward_functions import (
    r1_accuracy_reward_func,
    r1_int_reward_func,
    r1_strict_format_reward_func,
    r1_soft_format_reward_func,
    r1_count_xml,
)


def create_sample_dataset(tokenizer):
    """Create a sample dataset for GRPO training.
    
    Replace this with your actual dataset loading logic.
    """
    # Example reasoning dataset
    dataset = []
    
    # Math reasoning examples
    math_prompts = [
        "What is 25 + 17?",
        "Calculate 156 / 12",
        "Solve for x: 2x + 5 = 13",
        "What is the area of a circle with radius 5?",
        "Find the square root of 144",
    ]
    
    math_answers = [
        "42",
        "13", 
        "x = 4",
        "78.54",
        "12",
    ]
    
    # Add more diverse prompts for better training
    reasoning_prompts = [
        "Explain why the sky is blue.",
        "What causes seasons on Earth?",
        "How does photosynthesis work?",
        "Why do objects fall to the ground?",
        "What is the water cycle?",
    ]
    
    reasoning_answers = [
        "The sky appears blue because molecules in Earth's atmosphere scatter blue light from the sun more than other colors.",
        "Seasons are caused by Earth's tilted axis as it orbits the sun, changing the angle and intensity of sunlight.",
        "Photosynthesis converts light energy, water, and CO2 into glucose and oxygen in plant cells.",
        "Objects fall due to gravity, the force of attraction between masses.",
        "Water evaporates, forms clouds, precipitates, and returns to bodies of water in a continuous cycle.",
    ]
    
    # Combine all examples
    all_prompts = math_prompts + reasoning_prompts
    all_answers = math_answers + reasoning_answers
    
    # Create dataset tuples
    for prompt, answer in zip(all_prompts, all_answers):
        # Tokenize properly
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(answer)
        dataset.append((prompt_tokens, answer_tokens, prompt, answer, "reasoning"))
    
    return dataset


def load_dataset_from_file(dataset_path: str, tokenizer):
    """Load dataset from a JSON file.
    
    Expected format:
    [
        {
            "prompt": "Question or prompt text",
            "answer": "Expected answer",
            "type": "reasoning"  # optional
        },
        ...
    ]
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        prompt = item['prompt']
        answer = item['answer']
        task_type = item.get('type', 'general')
        
        # Tokenize the prompt and answer
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(answer)
        
        dataset.append((prompt_tokens, answer_tokens, prompt, answer, task_type))
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train GRPO with async generation")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the MLX inference server"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model (same as used by inference server)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to training dataset JSON file (optional, uses sample data if not provided)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to save LoRA adapters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of completions per prompt"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA adapter rank"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=10.0,
        help="LoRA adapter scale"
    )
    parser.add_argument(
        "--num-batches-ahead",
        type=int,
        default=2,
        help="Number of batches to generate ahead"
    )
    parser.add_argument(
        "--disable-async",
        action="store_true",
        help="Disable async generation (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Parse server URL to get host and port
    from urllib.parse import urlparse
    parsed_url = urlparse(args.server_url)
    server_host = parsed_url.hostname or "localhost"
    server_port = parsed_url.port or 8000
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)
    
    # Freeze base model
    model.freeze()
    
    lora_parameters = {
        "rank": args.lora_rank,
        "scale": args.lora_scale,
        "dropout": 0.0,
    }
    num_layers = len(model.layers)
    # Apply LoRA layers
    print(f"Applying LoRA layers with rank={args.lora_rank}...")
    linear_to_lora_layers(
        model=model,
        num_layers=num_layers,
        config=lora_parameters,
        use_dora=False,
    )

    print_trainable_parameters(model)

    
    # Load or create dataset
    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}...")
        train_dataset = load_dataset_from_file(args.dataset_path, tokenizer)
    else:
        print("Using sample dataset...")
        train_dataset = create_sample_dataset(tokenizer)
    
    # Split into train/val (90/10 split)
    split_idx = int(len(train_dataset) * 0.9)
    val_dataset = train_dataset[split_idx:]
    train_dataset = train_dataset[:split_idx]
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Configure training arguments
    training_args = GRPOTrainingArgs(
        # Basic training params
        batch_size=args.batch_size,
        iters=args.num_iters,
        steps_per_report=10,
        steps_per_eval=100,
        steps_per_save=500,
        adapter_file=f"{args.adapter_path}/adapters.safetensors",
        
        # GRPO specific params
        group_size=args.group_size,
        max_completion_length=1024,
        temperature=0.8,
        beta=0.1,  # KL penalty
        epsilon=0.001,  # PPO clipping
        
        # Async generation params
        enable_async_generation=not args.disable_async,
        num_batches_ahead=args.num_batches_ahead,
        async_generation_timeout=600.0,
        async_max_queue_size=5,
        num_iterations=1,  # Reuse each batch once
        inference_server_host=server_host,
        inference_server_port=server_port,
        
        # LoRA params
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=0.0,
        
        # Memory settings
        grad_checkpoint=True,  # Enable gradient checkpointing
        max_seq_length=2048,
    )
    
    # Define reward functions
    reward_functions = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ]
    
    # You can also define custom reward functions
    def custom_length_reward(prompts, completions, answer, types=None):
        """Reward shorter completions that contain the answer."""
        rewards = []
        for completion, expected in zip(completions, answer):
            if expected.lower() in completion.lower():
                # Reward inversely proportional to length
                reward = 1.0 / (1 + len(completion) / 100)
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards
    
    # Add custom reward function
    reward_functions.append(custom_length_reward)
    
    # Set reward weights (optional - defaults to equal weights)
    reward_weights = [1.0, 1.0, 0.5, 0.5, 0.5, 2.0]  # Higher weight for custom reward
    training_args.reward_weights = reward_weights
    
    # Create optimizer
    from mlx.optimizers import AdamW
    optimizer = AdamW(learning_rate=args.learning_rate)
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Async generation: {'Enabled' if training_args.enable_async_generation else 'Disabled'}")
    print(f"Inference server: {args.server_url}")
    print(f"Batch size: {training_args.batch_size}")
    print(f"Group size: {training_args.group_size}")
    print(f"Batches ahead: {training_args.num_batches_ahead}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Iterations: {training_args.iters}")
    print(f"Reward functions: {len(reward_functions)}")
    print("==============================\n")
    
    # Run training
    try:
        print("Starting GRPO training...")
        train_grpo(
            model=model,
            ref_model=None,  # Use same model as reference
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            reward_funcs=reward_functions,
            args=training_args,
            iterate_batches=iterate_grpo_batches,
        )
        
        print(f"\nTraining completed! Adapters saved to {args.adapter_path}")
        
        # Save final model config for easy loading
        save_config_path = Path(args.adapter_path) / "config.json"
        config = {
            "base_model": args.model_path,
            "adapter_path": args.adapter_path,
            "lora_rank": args.lora_rank,
            "lora_scale": args.lora_scale,
            "training_args": {
                "batch_size": training_args.batch_size,
                "group_size": training_args.group_size,
                "iterations": training_args.iters,
                "learning_rate": args.learning_rate,
            }
        }
        
        with open(save_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Saved training config to {save_config_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()