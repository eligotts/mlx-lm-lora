# %% [markdown]
# # Import necessary packages if not already installed

# %%
#!pip install mlx-lm-lora mlx-lm datasets
# Configure WandB - paste your API key when prompted
import wandb

# Set your WandB API key here
WANDB_API_KEY = os.getenv("WANDB_API_KEY")  # <-- Replace with your actual WandB API key

# Login to WandB
wandb.login(key=WANDB_API_KEY)

print("✅ WandB configured successfully!")

# %% [markdown]
# # Import your needed modules

# %%
from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo
from mlx_lm_lora.trainer.datasets import CacheDataset, GRPODataset
from mlx_lm_lora.utils import fuse_and_save_model

from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, HfApi

from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.utils import load, save_config

import mlx.optimizers as optim
import mlx.core as mx

from pathlib import Path
import re
import os


# %% [markdown]
# # Define the Args

# %%
hf_token = os.getenv("HF_TOKEN") # <-- Add you HF Token here

model_name = "Qwen/Qwen2-0.5B-Instruct-MLX"
user_name = "mlx-community"

adapter_path = "/Users/eligottlieb/Documents/mlx-lm-lora/examples/tests"
new_model_name = "new_model"
max_seq_length = 1028
num_layers = 12
lora_parameters = {"rank": 16, "dropout": 0.0, "scale": 10.0}

# dataset_name = "Goekdeniz-Guelmez/Big-Math-RL-Verified-MLX"
dataset_name = "ergotts/ethics_questions"

# %% [markdown]
# # Load the model

# %%

model, tokenizer = load(model_name)

# Verify vocab sizes match
model_vocab_size = model.args.vocab_size
# TokenizerWrapper uses vocab_size property
tokenizer_vocab_size = tokenizer.vocab_size
print(f"Model vocab size: {model_vocab_size}")
print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

if model_vocab_size != tokenizer_vocab_size:
    print(f"⚠️ WARNING: Model vocab size ({model_vocab_size}) doesn't match tokenizer vocab size ({tokenizer_vocab_size})")
    print("This may cause index out of range errors during generation.")

# %% [markdown]
# # Convert to LoRA

# %%
model.freeze()

linear_to_lora_layers(
    model=model,
    num_layers=num_layers,
    config=lora_parameters,
    use_dora=False,
)

print_trainable_parameters(model)

# %% [markdown]
# # Define the Optimizer

# %%
opt = optim.AdamW(learning_rate=1e-5)

# %% [markdown]
# # Load and Preprocess your Dataset using your custom Prompt Format

# %%
system_prompt = """You are an expert in ethical thinking. You are given a ethical question and you need to think about it and answer it.
You respond in the following format:
<thinking>
...
</thinking>
<answer>
...
</answer>"""

XML_COT_FORMAT = """<josie_thinks> {reasoning} </josie_thinks> <josie_answers> {answer} </josie_answers>"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<josie_answers>")[-1]
    answer = answer.split("</josie_answers>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        "system": system_prompt
    })
    return data

dataset = get_gsm8k_questions()


# Reward functions
def get_completion_content(completion):
    try:
        if isinstance(completion, str):
            return completion
        elif isinstance(completion, dict):
            return completion.get('content', '')
        elif isinstance(completion, list) and len(completion) > 0:
            first_item = completion[0]
            if isinstance(first_item, dict):
                return first_item.get('content', '')
            return str(first_item)
        return str(completion)
    except Exception:
        return ''

def get_prompt_content(prompt):
    try:
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, dict):
            return prompt.get('content', '')
        elif isinstance(prompt, list):
            last_item = prompt[-1]
            if isinstance(last_item, dict):
                return last_item.get('content', '')
            return str(last_item)
        return str(prompt)
    except Exception:
        return ''

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [get_completion_content(completion) for completion in completions]
    q = get_prompt_content(prompts[0])
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [get_completion_content(completion) for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<josie_thinks> .*? </josie_thinks> <josie_answers> .*? </josie_answers>\n$"
    responses = [get_completion_content(completion) for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<josie_thinks>.*?</josie_thinks><josie_answers>.*?</josie_answers>"
    responses = [get_completion_content(completion) for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<josie_thinks>") == 1:
        count += 0.125
    if text.count("</josie_thinks>") == 1:
        count += 0.125
    if text.count("<josie_answers>") == 1:
        count += 0.125
        count -= len(text.split("</josie_answers>")[-1])*0.001
    if text.count("</josie_answers>") == 1:
        count += 0.125
        count -= (len(text.split("</josie_answers>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [get_completion_content(completion) for completion in completions]
    return [count_xml(c) for c in contents]

# %%
print(dataset[0])

# %% [markdown]
# # 📦 Make the Dataset for the trainer

# %%
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.01, seed=42).values()

train_set = GRPODataset(train_dataset, tokenizer)
valid_set = GRPODataset(train_dataset, tokenizer)

# %%
# Import the reward function registry to register custom functions
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

# Register custom reward functions with specific names for JOSIE
@register_reward_function()
def josie_correctness_reward_func(prompts, completions, answer, types=None) -> list[float]:
    """Reward function for correctness - highest priority"""
    responses = [get_completion_content(completion) for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

@register_reward_function()
def josie_int_reward_func(prompts, completions, answer, types=None) -> list[float]:
    """Reward function for integer answers"""
    responses = [get_completion_content(completion) for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

@register_reward_function()
def josie_strict_format_reward_func(prompts, completions, answer, types=None) -> list[float]:
    """Reward function for strict XML formatting"""
    pattern = r"^<josie_thinks> .*? </josie_thinks> <josie_answers> .*? </josie_answers>\n$"
    responses = [get_completion_content(completion) for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

@register_reward_function()
def josie_soft_format_reward_func(prompts, completions, answer, types=None) -> list[float]:
    """Reward function for soft XML formatting"""
    pattern = r"<josie_thinks>.*?</josie_thinks><josie_answers>.*?</josie_answers>"
    responses = [get_completion_content(completion) for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

@register_reward_function()
def josie_xmlcount_reward_func(prompts, completions, answer, types=None) -> list[float]:
    """Reward function for XML tag counting"""
    contents = [get_completion_content(completion) for completion in completions]
    return [count_xml(c) for c in contents]

print("✅ Custom reward functions registered successfully!")
print("Available functions:", [
    "josie_correctness_reward_func",
    "josie_int_reward_func", 
    "josie_strict_format_reward_func",
    "josie_soft_format_reward_func",
    "josie_xmlcount_reward_func"
])

# %% [markdown]
# # Make the Adapter Folder and save the configs for loading later

# %%
# Import function to get custom reward functions
from mlx_lm_lora.trainer.grpo_reward_functions import get_reward_function

# Create your custom reward function list
custom_reward_functions = [
    get_reward_function("josie_correctness_reward_func"),
    get_reward_function("josie_int_reward_func"),
    get_reward_function("josie_strict_format_reward_func"), 
    get_reward_function("josie_soft_format_reward_func"),
    get_reward_function("josie_xmlcount_reward_func")
]

# Update weights to match your 5 custom functions
custom_reward_weights = [
    2.0,  # josie_correctness_reward_func - highest weight for correctness
    0.5,  # josie_int_reward_func - medium weight for integer answers
    1.0,  # josie_strict_format_reward_func - standard weight for strict formatting
    0.8,  # josie_soft_format_reward_func - slightly lower weight for soft formatting  
    0.3   # josie_xmlcount_reward_func - lower weight for XML tag counting
]

print("✅ Custom reward functions loaded:")
for i, func in enumerate(custom_reward_functions):
    print(f"  {i+1}. {func.__name__} (weight: {custom_reward_weights[i]})")


# %%
args = {
    "lora_parameters": lora_parameters,
    "num_layers": num_layers,
}

adapter_path = Path(adapter_path)
adapter_path.mkdir(parents=True, exist_ok=True)

adapter_file = adapter_path / "adapters.safetensors"
save_config(args, adapter_path / "adapter_config.json")

# %% [markdown]
# # Start training

# %%
# Import WandBCallback
from mlx_lm.tuner.callbacks import WandBCallback

# Define custom reward weights if you want to weight them differently
# The weights correspond to the 5 default reward functions in order
# custom_reward_weights = [
#     2.0,  # r1_accuracy_reward_func - highest weight for correctness
#     0.5,  # r1_int_reward_func - medium weight for integer answers
#     1.0,  # r1_strict_format_reward_func - standard weight for strict formatting
#     0.8,  # r1_soft_format_reward_func - slightly lower weight for soft formatting  
#     0.3   # r1_count_xml - lower weight for XML tag counting
# ]

# Configure WandB callback
wandb_callback = WandBCallback(
    project_name="grpo-test",  # Your WandB project name
    log_dir=str(adapter_path),  # Directory for logs
    config={
        "model": model_name,
        "batch_size": 1,
        "iters": 200,
        "learning_rate": 1e-5,
        "num_layers": num_layers,
        "lora_rank": lora_parameters["rank"],
        "max_seq_length": max_seq_length,
        "beta": 0.9,
        "group_size": 4,
        "gradient_accumulation_steps": 5,
        "reward_weights": custom_reward_weights,
    }
)

train_grpo(
    model=model,
    ref_model=None,  # Use None to use the same model as reference
    tokenizer=tokenizer,  # Add the missing tokenizer argument
    optimizer=opt,
    train_dataset=CacheDataset(train_set),
    val_dataset=CacheDataset(valid_set),
    args=GRPOTrainingArgs(
        batch_size=1,
        iters=200,
        val_batches=1,
        steps_per_report=10, #20,
        steps_per_eval=50, # 50,
        steps_per_save=100, # 50,
        adapter_file=adapter_file,
        max_seq_length=max_seq_length,
        grad_checkpoint=True,
        gradient_accumulation_steps=5,
        beta=0.9,
        group_size=4,
        epsilon=1e-4,
        epsilon_high=None,
        max_completion_length=1028,
        reward_weights=custom_reward_weights,  # Use this instead of reward_scaling
    ),
    reward_funcs=custom_reward_functions,  # Pass the custom reward functions
    training_callback=wandb_callback  # Pass the WandB callback here
)

# %% [markdown]
# # Fuse the model with the trained adapters and save the new model

# %%
# fuse_and_save_model(
#     model=model,
#     tokenizer=tokenizer,
#     save_path=new_model_name,
#     adapter_path=adapter_path,
#     de_quantize=False,
#     export_gguf=False,
#     gguf_path=f"{new_model_name}/model.gguf",
# )

# %% [markdown]
# # Create the README

# %%
readme_file = f"""---
tags:
- mlx
- lora
- text-generation
- fine-tuning
base_model: {model_name}
pipeline_tag: text-generation
---

# LoRA Fine-Tuned Model: `{user_name}/{new_model_name}`

This model is a LoRA fine-tuned version `{model_name}`, with the [`mlx-lm-lora`](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) training package on Apple Silicon using MLX.

---

## 🧾 Model Details

- **Model name:** {new_model_name}
- **Base model:** {model_name}
- **Fine-tuning method:** GRPO
- **Training package:** [`MLX-LM-LORA`](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- **Model type:** {model.args.model_type}
- **Author:** None

---

## 💡 Recommended System Prompt

```text
{system_prompt}
```
"""

new_readme_path = f"{new_model_name}/README.md"
with open(new_readme_path, "w") as new_readme_file:
    new_readme_file.write(readme_file)

# %% [markdown]
# # Upload it to HugginFace

# %%
# api = HfApi(token=hf_token)
# create_repo(
#   repo_id = f"{user_name}/{new_model_name}",
#   repo_type="model",
#   exist_ok=True,
#   token=hf_token,
#   private=True
# )
# api.upload_folder(
#   folder_path=new_model_name,
#   repo_id=f"{user_name}/{new_model_name}",
#   token=hf_token,
#   commit_message="Initial Commit"
# )


