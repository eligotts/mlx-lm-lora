# Async GRPO Training with MLX RL Inference Server

This guide explains how to run GRPO training with asynchronous generation using the MLX RL inference server.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install mlx mlx-lm aiohttp
   ```

2. **Start the MLX Inference Server**
   ```bash
   # Start the server with your base model
   python -m mlx_lm.inference_server --model /path/to/your/model
   ```
   
   The server will start on `http://localhost:8000` by default.

## Running Training

### Basic Usage

```bash
python train_async_grpo.py --model-path /path/to/your/model
```

### Full Command Line Options

```bash
python train_async_grpo.py \
    --server-url http://localhost:8000 \
    --model-path /path/to/your/model \
    --dataset-path example_dataset.json \
    --adapter-path ./adapters \
    --batch-size 4 \
    --group-size 4 \
    --num-iters 1000 \
    --learning-rate 1e-5 \
    --lora-rank 16 \
    --lora-scale 10.0 \
    --num-batches-ahead 2
```

### Command Line Arguments

- `--server-url`: URL of the MLX inference server (default: http://localhost:8000)
- `--model-path`: Path to the model (must be same as inference server)
- `--dataset-path`: Path to training dataset JSON file (optional)
- `--adapter-path`: Path to save LoRA adapters (default: ./adapters)
- `--batch-size`: Training batch size (default: 4)
- `--group-size`: Number of completions per prompt (default: 4)
- `--num-iters`: Number of training iterations (default: 1000)
- `--learning-rate`: Learning rate (default: 1e-5)
- `--lora-rank`: LoRA adapter rank (default: 16)
- `--lora-scale`: LoRA adapter scale (default: 10.0)
- `--num-batches-ahead`: Number of batches to generate ahead (default: 2)
- `--disable-async`: Disable async generation for debugging

## Dataset Format

The training script expects a JSON file with the following format:

```json
[
  {
    "prompt": "What is 15 + 27?",
    "answer": "15 + 27 = 42",
    "type": "math"
  },
  {
    "prompt": "Explain photosynthesis",
    "answer": "Photosynthesis is the process...",
    "type": "reasoning"
  }
]
```

See `example_dataset.json` for a complete example.

## How It Works

1. **Async Generation Pipeline**: The training script generates batches asynchronously on the inference server while processing previous batches on the training machine.

2. **Weight Synchronization**: LoRA adapter weights are automatically synchronized to the inference server when new generations are needed.

3. **Batch Management**: The system maintains a pipeline of batches, generating `num_batches_ahead` batches in advance to maximize GPU utilization.

4. **Chat Template Formatting**: Prompts are automatically formatted with the model's chat template before being sent to the inference server.

## Example Training Session

```bash
# Terminal 1: Start the inference server
python -m mlx_lm.inference_server --model mlx-community/Qwen3-0.6B-4bit

# Terminal 2: Run training
python train_async_grpo.py \
    --model-path mlx-community/Qwen3-0.6B-4bit \
    --dataset-path example_dataset.json \
    --batch-size 4 \
    --group-size 4 \
    --num-iters 500
```

## Monitoring Training

The training script will output:
- Training loss and metrics every 10 steps
- Validation loss every 100 steps
- Adapter checkpoints every 500 steps

## Using Trained Adapters

After training, you can load the trained LoRA adapters:

```python
from mlx_lm import load

model, tokenizer = load(
    "path/to/base/model",
    adapter_path="./adapters"
)
```

## Troubleshooting

1. **Connection Error**: Make sure the inference server is running and accessible at the specified URL.

2. **Out of Memory**: Reduce `batch_size` or `group_size`.

3. **Slow Generation**: Decrease `num_batches_ahead` to reduce memory usage on the inference server.

4. **Training Instability**: Adjust `beta` (KL penalty) and `epsilon` (PPO clipping) parameters.