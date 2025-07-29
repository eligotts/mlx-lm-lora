import sys
from mlx_lm.utils import load
from mlx_lm.generate import generate
from mlx_lm_lora.trainer.grpo_trainer import make_safe_sampler

# Model and tokenizer setup (from grpo_2.py)
# model_name = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"
model_name = "willcb/Qwen3-4B"
model, tokenizer = load(model_name)

# Define your prompt here
prompt = "Hello, what are you?"

# Create the safe sampler
sampler = make_safe_sampler(
    tokenizer=tokenizer,
    temperature=0.8,
    top_p=1.0,
    min_p=0.0,
    min_tokens_to_keep=1,
    top_k=0,
    xtc_probability=0.0,
    xtc_threshold=0.0,
    xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
)

# Generate a completion
completion = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=128,
    verbose=True,
    sampler=sampler,
)

print("\n=== Generated Output ===\n")
print(completion) 