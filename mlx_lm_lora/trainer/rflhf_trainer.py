from dataclasses import dataclass, field
from pathlib import Path
import time

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.generate import generate

from .sft_trainer import SFTTrainingArgs, grad_checkpoint

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm


@dataclass
class RLHFTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, 
        metadata={"help": "KL penalty coefficient for RLHF training."}
    )
    reward_model_path: str = field(
        default=None,
        metadata={"help": "Path to reward model weights."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={"help": "Path to reference model weights."}
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max tokens to generate per prompt."}
    )

def compute_kl_penalty(logits_policy, logits_ref, masks):
    policy_probs = nn.softmax(logits_policy, axis=-1)
    ref_probs = nn.softmax(logits_ref, axis=-1)
    
    kl_div = policy_probs * (mx.log(policy_probs) - mx.log(ref_probs))
    kl_div = mx.sum(kl_div, axis=-1)
    return mx.sum(kl_div * masks, axis=-1)

def generate_responses(model, tokenizer, prompts, max_tokens):
    responses = []
    for prompt in prompts:
        response = generate(model, tokenizer, prompt, max_tokens)
        responses.append(response)
    return responses

def rlhf_loss(
    policy_logits: mx.array,
    ref_logits: mx.array,
    rewards: mx.array,
    masks: mx.array,
    beta: float,
):
    # Compute log probabilities for actual tokens
    labels = mx.argmax(policy_logits, axis=-1)
    policy_log_probs = -nn.losses.cross_entropy(policy_logits, labels, reduction='none')
    ref_log_probs = -nn.losses.cross_entropy(ref_logits, labels, reduction='none')
    
    # Compute KL divergence per token
    kl_div = policy_log_probs - ref_log_probs
    
    # Sum KL over sequence and average over batch
    kl_penalty = (kl_div * masks).sum(axis=-1)
    
    # Policy gradient loss
    advantages = rewards - beta * kl_penalty
    loss = -advantages * (policy_log_probs * masks).sum(axis=-1)
    
    # Normalize by token count
    token_count = masks.sum()
    loss = loss.sum() / token_count
    
    # Compute metrics
    metrics = {
        "rewards": mx.mean(rewards),
        "kl_penalty": mx.mean(kl_penalty),
        "advantages": mx.mean(advantages),
        "policy_logps": mx.mean(policy_log_probs),
        "ref_logps": mx.mean(ref_log_probs)
    }
    
    mx.clear_cache()
    return loss, token_count, metrics


def iterate_rlhf_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["prompt"]))
    
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")
    
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    
    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            prompts = [x["prompt"] for x in batch]
            
            # Pad prompts
            max_prompt_len = min(max(len(p) for p in prompts), max_seq_length)
            prompt_arr = np.zeros((len(prompts), max_prompt_len), dtype=np.int32)
            prompt_masks = np.zeros((len(prompts), max_prompt_len), dtype=np.float32)
            
            for j, p in enumerate(prompts):
                prompt_len = min(len(p), max_prompt_len)
                prompt_arr[j, :prompt_len] = p[:prompt_len]
                prompt_masks[j, :prompt_len] = 1.0
                
            yield mx.array(prompt_arr), mx.array(prompt_masks)
        
        if not train:
            break