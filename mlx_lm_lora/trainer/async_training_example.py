"""Example of how to use async GRPO training with train-one-off methodology."""

from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo
from mlx_lm_lora.trainer.example_inference_client import MLXInferenceClient, MockInferenceClient
from mlx_lm_lora.trainer.grpo_reward_functions import (
    r1_accuracy_reward_func,
    r1_int_reward_func,
    r1_strict_format_reward_func,
)


def train_with_async_generation():
    """Example of training with async generation enabled.
    
    Prerequisites:
    1. Start the MLX inference server with your base model:
       python -m mlx_lm.inference_server --model <model_path>
       
    2. The server should be running on http://localhost:8000
    
    3. The model must support LoRA adapters for weight updates
    """
    
    # Configure training arguments with async enabled
    args = GRPOTrainingArgs(
        # Standard GRPO arguments
        batch_size=4,
        group_size=4,
        max_completion_length=512,
        temperature=0.8,
        beta=0.1,
        epsilon=0.001,
        gradient_accumulation_steps=4,
        
        # Async-specific arguments
        enable_async_generation=True,
        num_batches_ahead=2,  # Generate 2 batches ahead
        async_generation_timeout=600.0,  # 10 minute timeout
        async_max_queue_size=5,  # Max 5 batches in queue
        num_iterations=1,  # Use each batch once
        
        # Inference server configuration (MLX RL inference server)
        inference_server_host="localhost",
        inference_server_port=8000,  # Default MLX inference server port
        
        # LoRA configuration (if using LoRA)
        lora_rank=16,
        lora_scale=10.0,
        lora_dropout=0.0,
        
        # Other training arguments
        iters=1000,
        steps_per_report=10,
        steps_per_eval=100,
        steps_per_save=500,
        adapter_file="adapters.safetensors",
    )
    
    # Define reward functions
    reward_funcs = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
    ]
    
    # Assume model, tokenizer, optimizer, and datasets are already initialized
    # model = ...
    # tokenizer = ...
    # optimizer = ...
    # train_dataset = ...
    # val_dataset = ...
    
    # Train with async generation
    # train_grpo(
    #     model=model,
    #     ref_model=None,  # Use same model as reference
    #     tokenizer=tokenizer,
    #     optimizer=optimizer,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     reward_funcs=reward_funcs,
    #     args=args,
    # )


def test_with_mock_client():
    """Example using mock client for testing without real inference server."""
    
    args = GRPOTrainingArgs(
        # Enable async with mock client
        enable_async_generation=True,
        num_batches_ahead=1,
        num_iterations=2,  # Reuse each batch twice
        gradient_accumulation_steps=2,
        
        # Other args...
        batch_size=2,
        group_size=2,
        max_completion_length=128,
    )
    
    # In the training function, you would use MockInferenceClient:
    # 
    # if args.enable_async_generation:
    #     # Use mock client for testing
    #     inference_client = MockInferenceClient(model)
    #     
    #     async_generator = AsyncBatchGenerator(
    #         inference_client=inference_client,
    #         num_batches_ahead=args.num_batches_ahead,
    #         ...
    #     )


def understanding_batch_pipeline():
    """Explanation of how the batch pipeline works."""
    
    # The async training pipeline maintains a queue of batches being generated
    # while the training loop processes previously generated batches.
    
    # Key concepts:
    
    # 1. Generation frequency:
    #    New batches are only generated every (gradient_accumulation_steps * num_iterations) steps
    #    This controls how often we need fresh generations
    
    # 2. Pipeline depth:
    #    num_batches_ahead controls how many batches are generated in advance
    #    Higher values = better GPU utilization but more staleness
    
    # 3. Weight synchronization:
    #    Weights are synced to inference server only when new generations are needed
    #    This minimizes overhead while preventing excessive drift
    
    # 4. Batch reuse:
    #    num_iterations > 1 allows reusing the same generations multiple times
    #    This amortizes generation cost but increases staleness
    
    # Example timeline with num_batches_ahead=2, num_iterations=1, gradient_accumulation_steps=4:
    
    # Step 0: Submit batches 0, 1, 2 for generation, retrieve batch 0
    # Steps 1-3: Use batch 0 (no new generation)
    # Step 4: Submit batch 3, retrieve batch 1, sync weights
    # Steps 5-7: Use batch 1
    # Step 8: Submit batch 4, retrieve batch 2, sync weights
    # ... and so on
    
    print("See comments in code for detailed explanation")


if __name__ == "__main__":
    # Run examples
    understanding_batch_pipeline()
    
    # Uncomment to run actual training
    # train_with_async_generation()
    # test_with_mock_client()