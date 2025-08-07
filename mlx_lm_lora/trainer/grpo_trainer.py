from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import time
import asyncio
import aiohttp
import json
import tempfile
import shutil

from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.tuner.callbacks import TrainingCallback

from .sft_trainer import SFTTrainingArgs, average_gradients, grad_checkpoint

from mlx_lm.models import cache
from mlx_lm.generate import generate, make_sampler
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)


def make_safe_sampler(tokenizer, temperature,
                      top_p=1.0, min_p=0.0, min_tokens_to_keep=1,
                      top_k=0, xtc_probability=0.0, xtc_threshold=0.0,
                      xtc_special_tokens=None):

    base_sampler = make_sampler(
        temperature,
        top_p,
        min_p,
        min_tokens_to_keep,
        top_k,
        xtc_probability,
        xtc_threshold,
        xtc_special_tokens,
    )
    vocab_size = tokenizer.vocab_size        # <= 151 665 for Qwen-2.5-0.5B

    # Pre-compute a Boolean mask once; re-use every step.
    # Shape: (1, 1, vocab_size_model) so it broadcasts on batches/time.
    _mask = None                             # closed-over cell

    def safe_sampler(logits):
        nonlocal _mask
        if logits.shape[-1] > vocab_size:
            if _mask is None or _mask.shape[-1] != logits.shape[-1]:
                _mask = mx.arange(logits.shape[-1]) >= vocab_size  # True where invalid
            # functional masking – makes a *new* array, original logits untouched
            logits = mx.where(_mask, float('-inf'), logits)
        return base_sampler(logits)

    return safe_sampler


@dataclass
class GRPOTrainingArgs(SFTTrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    epsilon_high: float = field(
        default=None, metadata={"help": "For DAPO Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument epsilon."}
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    grpo_loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Type of loss to use for GRPO. Supported: 'grpo', 'bnpo', 'dr_grpo'."
        }
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )
    importance_sampling_level: str = field(
        default=None,
        metadata={
            "help": "importance_sampling_level (`str`, *optional*, defaults to None): "
                "Controls whether importance sampling ratios are computed at the 'token' or 'sequence' level. "
                "keeps the raw per-token log-probability ratios (one weight per token).  'sequence' averages the "
                "log-probability ratios across valid tokens to produce a single ratio per sequence. The "
                "GSPO paper https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more "
                "stable training and better alignment with  sequence-level rewards.."
        },
    )
    inference_server_url: str = field(
        default="http://10.0.0.180:8000",
        metadata={
            "help": "URL of the inference server for GRPO generation and adapter updates."
        },
    )
    upload_adapters_to_server: bool = field(
        default=True,
        metadata={
            "help": "Whether to upload adapter weights to the inference server after each update."
        },
    )


def get_per_token_logps(model: nn.Module, inputs, lengths):
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps

async def send_single_request(
    session,
    base_url: str,
    prompt: str,
    tokenizer,
    max_tokens: int,
    temperature: float,
    request_id: str,
    step_id: Optional[int] = None,
):
    """Send a single request to the inference server."""
    # Format prompt for chat template
    prompt_fm = [{"role": "user", "content": prompt}]
    prompt_fm = tokenizer.apply_chat_template(prompt_fm, add_generation_prompt=True, tokenize=False)
    
    payload = {
        "prompt": prompt_fm,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": 1,  # Request logprobs for GRPO
        "step_id": step_id or 100,
        "request_id": request_id
    }
    
    try:
        async with session.post(f"{base_url}/generate", json=payload) as resp:
            result = await resp.json()
            completion_text = result.get('text', '')
            # Truncate text for display if it's too long
            display_text = completion_text[:100] + "..." if len(completion_text) > 100 else completion_text
            display_text = display_text.replace('\n', ' ')  # Replace newlines for cleaner display
            print(f"✓ Received {request_id}: {len(result.get('completion_ids', []))} tokens | Text: '{display_text}'")
            return result
    except Exception as e:
        print(f"⚠️ Error calling inference server for {request_id}: {e}")
        # Return empty result on error
        return {
            "completion_ids": [],
            "text": "",
            "finish_reason": "error"
        }


async def generate_async_batch(
    session,
    base_url: str,
    prompts: List[str],
    tokenizer,
    max_tokens: int,
    temperature: float,
    step_id: Optional[int] = None,
):
    """Send batch requests to inference server in parallel and collect responses."""
    print(f"\n📤 Sending batch of {len(prompts)} prompts to inference server...")
    
    # Create all requests as coroutines
    tasks = []
    for idx, prompt in enumerate(prompts):
        request_id = f"grpo-{idx}-{step_id or 0}"
        task = send_single_request(
            session=session,
            base_url=base_url,
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            step_id=step_id,
        )
        tasks.append(task)
    
    # Execute all requests in parallel
    print(f"⏳ Waiting for all {len(tasks)} requests to complete...")
    results = await asyncio.gather(*tasks)
    print(f"✅ Batch complete! Received {len(results)} responses\n")
    
    return results


async def upload_adapter_weights(
    session,
    base_url: str,
    model,
    adapter_config: dict,
    expected_policy_version: int,
    step_id: int = None,
):
    """Upload adapter weights to the inference server and verify version consistency."""
    print(f"📤 Uploading adapter weights to server (step {step_id}, expecting version {expected_policy_version})...")
    
    try:
        # Extract adapter weights from model
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        
        # Create temporary directory for adapter files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save adapter weights to temporary file
            weights_path = temp_path / "adapters.safetensors"
            mx.save_safetensors(str(weights_path), adapter_weights)
            
            # Save adapter config to temporary file
            config_path = temp_path / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            # Prepare files for upload
            with open(weights_path, "rb") as weights_file:
                weights_data = weights_file.read()
            with open(config_path, "rb") as config_file:
                config_data = config_file.read()
            
            # Create multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field('adapter_weights', weights_data,
                              filename='adapters.safetensors',
                              content_type='application/octet-stream')
            form_data.add_field('adapter_config', config_data,
                              filename='adapter_config.json',
                              content_type='application/json')
            
            # Upload to server
            async with session.post(f"{base_url}/upload_adapter", data=form_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    server_policy_version = result.get('policy_version')
                    
                    # Assert versions match
                    if server_policy_version != expected_policy_version:
                        print(f"⚠️ WARNING: Policy version mismatch! Expected {expected_policy_version}, got {server_policy_version}")
                        print(f"   This may cause inconsistent behavior. Check server state.")
                    else:
                        print(f"✅ Adapter uploaded successfully! Policy version confirmed: {server_policy_version}")
                    
                    return server_policy_version
                else:
                    error = await resp.text()
                    print(f"❌ Failed to upload adapter: {error}")
                    raise RuntimeError(f"Adapter upload failed with status {resp.status}")
                    
    except Exception as e:
        print(f"❌ Error uploading adapter weights: {e}")
        raise


def generate_grpo(
    model: nn.Module,
    tokenizer,
    prompt_tokens,
    max_tokens: int,
    group_size: int,
    temperature: float,
    batch_size: int,
    end_token: str = "</answer>",
    inference_server_url: str = "http://10.0.0.180:8000",
    step_id: Optional[int] = None
):
    model.eval()
    all_completions = []
    all_completion_texts = []
    all_logprobs = []
    batch_indices = []

    total_samples = len(prompt_tokens)
    print(f"\n🚀 Starting GRPO generation for {total_samples} prompts with group_size={group_size}")
    print(f"   Server: {inference_server_url}")
    print(f"   Max tokens per generation: {max_tokens}")
    print(f"   Total requests to make: {total_samples * group_size}")

    async def run_async_generation():
        async with aiohttp.ClientSession() as session:
            nonlocal all_completions, all_completion_texts, batch_indices
            
            for i in range(0, total_samples, batch_size):
                current_batch_size = min(batch_size, total_samples - i)
                batch_prompts = prompt_tokens[i : i + current_batch_size]
                
                print(f"\n📦 Processing batch {i//batch_size + 1}: prompts {i} to {i + current_batch_size - 1}")
                
                # Collect all prompts for this batch
                batch_requests = []
                for j, prompt in enumerate(batch_prompts):
                    for k in range(group_size):
                        prompt_text = tokenizer.decode(prompt)
                        batch_requests.append((prompt_text, i + j))
                
                # Extract prompts and indices
                prompts_to_send = [req[0] for req in batch_requests]
                prompt_indices = [req[1] for req in batch_requests]
                
                print(f"   Batch contains {len(prompts_to_send)} total requests ({current_batch_size} prompts × {group_size} group_size)")
                
                # Send batch to inference server - ALL IN PARALLEL!
                results = await generate_async_batch(
                    session=session,
                    base_url=inference_server_url,
                    prompts=prompts_to_send,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    step_id=step_id,
                )
                
                # Process results
                successful_results = 0
                token_counts = []
                for result, prompt_idx in zip(results, prompt_indices):
                    try:
                        completion_ids = result.get("completion_ids", [])
                        completion_text = result.get("text", "")
                        logprobs = result.get("logprobs", [])
                        
                        if completion_ids:
                            successful_results += 1
                            token_counts.append(len(completion_ids))
                        
                        if end_token and completion_text:
                            # Remove end token if present
                            end_sequence = tokenizer.encode(end_token)
                            if len(completion_ids) >= len(end_sequence) and completion_ids[-len(end_sequence):] == end_sequence:
                                completion_ids = completion_ids[:-len(end_sequence)]

                                # also remove the logprobs for the end token
                                logprobs = logprobs[:-len(end_sequence)]

                                
                        
                        completion_ids = mx.array(completion_ids)
                        all_completions.append(mx.stop_gradient(completion_ids))
                        all_completion_texts.append(completion_text)
                        all_logprobs.append(mx.array(logprobs))
                        batch_indices.append(prompt_idx)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing result: {e}")
                        # Handle error by adding empty completion
                        all_completions.append(mx.array([]))
                        all_completion_texts.append("")
                        all_logprobs.append(mx.array([]))
                        batch_indices.append(prompt_idx)
                
                if token_counts:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    min_tokens = min(token_counts)
                    max_tokens_generated = max(token_counts)
                    print(f"   ✅ Batch processed: {successful_results}/{len(results)} successful")
                    print(f"      Token stats: min={min_tokens}, max={max_tokens_generated}, avg={avg_tokens:.1f}")
                else:
                    print(f"   ✅ Batch processed: {successful_results}/{len(results)} successful completions")
    
    # Run async generation
    asyncio.run(run_async_generation())
    
    print(f"\n🎉 GRPO generation complete! Generated {len(all_completions)} completions total")
    
    mx.clear_cache()
    return all_completions, all_completion_texts, all_logprobs, batch_indices


def grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    completions=None,
    completion_texts=None,
    completion_logprobs=None,
    batch_indices=None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon: float = 1e-4,
    epsilon_high: float = None,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
    importance_sampling_level: str = "token",
    grpo_loss_type: str = "grpo",
):
    prompt_tokens, _, prompt_text, answer_text, type_info = batch

    if (
        completions is not None
        and completion_texts is not None
        and batch_indices is not None
    ):
        all_completions = completions
        all_completion_texts = completion_texts
        all_logprobs = completion_logprobs if completion_logprobs is not None else []
        batch_indices = batch_indices
    else:
        all_completions, all_completion_texts, all_logprobs, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size,
        )

    if not all_completions:
        raise ValueError(
            "No completions were generated. Please check your model and inputs."
        )

    expanded_answers = []
    expanded_prompts = []
    expanded_types = []
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_answers.append(answer_text[prompt_idx])
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_types.append(type_info[prompt_idx] if type_info is not None else None)

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        completion_tensor = mx.array(completion_ids.tolist())
        padding_length = max_length - completion_tensor.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate(
                [mx.ones_like(completion_tensor), mx.zeros_like(padding)]
            )
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)

    # old way:
    # token_log_probs = get_per_token_logps(model, inputs, lengths)
    # mx.eval(token_log_probs)

    # if ref_model is None:
    #     ref_token_log_probs = token_log_probs
    # else:
    #     ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths)
    #     mx.eval(ref_token_log_probs)

    # max_len = max(x.shape[0] for x in token_log_probs)
    # padded_log_probs = []
    # padded_ref_log_probs = []

    # for i in range(len(token_log_probs)):
    #     seq_len = token_log_probs[i].shape[0]
    #     padding = mx.zeros((max_len - seq_len,))

    #     padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
    #     padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    # old_token_log_probs = mx.stack(padded_log_probs)
    # old_ref_token_log_probs = mx.stack(padded_ref_log_probs)

    # new way:

    # Use logprobs from server if available, otherwise compute them
    if all_logprobs and len(all_logprobs) > 0 and all(lp is not None and lp.size > 0 for lp in all_logprobs):
        # Server provided logprobs - pad them to match completion padding
        print(f"   Using server-provided logprobs for {len(all_logprobs)} completions")
        
        # remove the last logprob for each sequence
        token_log_probs = [lp[:-1] for lp in all_logprobs]
    else:
        # Fallback to computing logprobs locally
        print(f"   Computing logprobs locally (server didn't provide them)")
        token_log_probs = get_per_token_logps(model, inputs, lengths)
        mx.eval(token_log_probs)


    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        # For reference model, we always need to compute locally using the padded inputs
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths)
        mx.eval(ref_token_log_probs)

    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)
    

    # At this point:
    # - token_log_probs is already a 2D padded tensor (from server or local computation)
    # - ref_token_log_probs is also a 2D padded tensor

    all_func_rewards = []
    for reward_func in reward_funcs:
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types
        )
        if raw_rewards is None:
            processed_rewards = [float('nan')] * len(all_completion_texts)
        else:
            processed_rewards = [float(r) if r is not None else float('nan') for r in raw_rewards]
        func_rewards = mx.array(processed_rewards)
        all_func_rewards.append(func_rewards)

    rewards = mx.stack(all_func_rewards, axis=1)

    all_nan_rows = mx.all(mx.isnan(rewards), axis=1)
    if mx.any(all_nan_rows):
        nan_row_idx = mx.argmax(all_nan_rows).item()
        warning_msg = (
            f"All reward functions returned None for prompt: {expanded_prompts[nan_row_idx]}, "
            f"completion: {all_completion_texts[nan_row_idx]}, "
            f"answer: {expanded_answers[nan_row_idx]}. "
            "Please ensure that at least one reward function returns a valid reward."
        )
        raise RuntimeError(warning_msg)

    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)

    valid_reward_mask = ~mx.isnan(rewards)
    rewards_no_nan = mx.where(valid_reward_mask, rewards, mx.zeros_like(rewards))
    rewards = (rewards_no_nan * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    num_unique_prompts = len(unique_prompt_indices)

    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards)
            std_reward = mx.std(prompt_rewards)
            indices = [
                j
                for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards[j] - mean_reward) / (
                    std_reward + 1e-4
                )
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    # Compute KL divergence using Schulman's approximator
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    )

    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Compute log ratio for importance sampling
    log_ratio = token_log_probs - mx.stop_gradient(ref_token_log_probs)

    # Apply importance sampling based on level
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        # Average log ratio over sequence length for each sequence
        sequence_log_ratio = (log_ratio * length_mask).sum(axis=1) / mx.maximum(length_mask.sum(axis=1), 1.0)
        log_importance_weights = mx.expand_dims(sequence_log_ratio, axis=1)
    elif importance_sampling_level is None or importance_sampling_level == "none":
        log_importance_weights = mx.zeros_like(log_ratio)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. "
            "Possible values are 'token', 'sequence', or None."
        )

    # Calculate importance weights
    coef_1 = mx.exp(log_importance_weights)

    # Apply PPO like clipping
    epsilon_high = epsilon_high if epsilon_high else epsilon
    coef_2 = mx.clip(coef_1, 1 - epsilon, 1 + epsilon_high)

    # Track clipping metrics
    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages.reshape(-1, 1) < 0)
    is_high_clipped = (coef_1 > 1 + epsilon_high) & (
        advantages.reshape(-1, 1) > 0
    )
    is_region_clipped = is_low_clipped | is_high_clipped


    # Calculate both unclipped and clipped objectives
    unclipped_obj = coef_1 * advantages.reshape(-1, 1)
    clipped_obj = coef_2 * advantages.reshape(-1, 1)

    # Take the minimum (pessimistic bound)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)

    # Add KL penalty if beta is non-zero
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div

    
    if grpo_loss_type == "grpo":
        loss = (per_token_loss * length_mask).sum() / length_mask.sum()
    elif grpo_loss_type == "bnpo":
        loss = (per_token_loss * length_mask).sum() / mx.maximum(length_mask.sum(), 1.0)
    elif grpo_loss_type == "dr_grpo":
        loss = (per_token_loss * length_mask).sum() / (per_token_loss.shape[0] * max_tokens)
    else:
        raise ValueError(f"Unknown loss type: {grpo_loss_type}")

    # Calculate mean KL divergence for metrics
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()

    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        valid_mask = ~mx.isnan(mx.array([reward if reward is not None else float('nan') for reward in raw_rewards]))
        valid_rewards = mx.array([reward for reward in raw_rewards if reward is not None and not mx.isnan(reward)])
        if len(valid_rewards) > 0:
            reward_metrics[f"{func_name}_mean"] = mx.mean(valid_rewards)
            reward_metrics[f"{func_name}_std"] = mx.std(valid_rewards) if len(valid_rewards) > 1 else mx.zeros(1)
            reward_metrics[f"{func_name}_coverage"] = valid_mask.sum() / len(raw_rewards)
        else:
            reward_metrics[f"{func_name}_mean"] = float('nan')
            reward_metrics[f"{func_name}_std"] = float('nan')
            reward_metrics[f"{func_name}_coverage"] = 0.0

    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array(
        [
            mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
            for rewards in rewards_by_prompt
        ]
    )

    metrics = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards),
        "grouped_rewards_mean": mx.mean(grouped_rewards_mean),
        "grouped_rewards_std": mx.mean(grouped_rewards_std),
        "kl": mean_kl,
        "average_generated_tokens": len(all_completion_texts) // len(batch_indices),
        "clip_ratio_low": (
            (is_low_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_high": (
            (is_high_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_total": (
            (is_region_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        **reward_metrics,
    }

    mx.clear_cache()

    return loss, length_mask.sum(axis=1).sum(), metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    has_types = isinstance(dataset[0], tuple) and len(dataset[0]) == 5

    if not dataset or not isinstance(dataset[0], tuple) or (not has_types and len(dataset[0]) != 4):
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str[, type]) tuples"
        )

    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]

            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]
            types = [item[4] for item in current_batch] if has_types else None

            yield prompts_tokens, answers_tokens, prompts_text, answers_text, types

        if not train:
            break


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    epsilon_high: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    grpo_loss_type: str = "grpo",
    importance_sampling_level: str = "token",
):
    all_losses = 0
    ntokens = 0
    all_metrics = None

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks, metrics = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            ref_model=ref_model,
            temperature=temperature,
            max_tokens=max_tokens,
            importance_sampling_level=importance_sampling_level,
            grpo_loss_type=grpo_loss_type,
        )

        all_losses += losses * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, ntokens, avg_metrics


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    tqdm.write(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]
    
    # Initialize policy version tracking - always start at 1
    current_policy_version = 1
    
    # Prepare adapter config and upload initial weights if enabled
    adapter_config = None
    if args.upload_adapters_to_server:
        adapter_config = {
            "model_type": model.model_type if hasattr(model, 'model_type') else "unknown",
            "num_layers": len(model.layers) if hasattr(model, 'layers') else 0,
            "lora_parameters": {
                "rank": getattr(args, 'lora_rank', 16),
                "dropout": getattr(args, 'lora_dropout', 0.0),
                "scale": getattr(args, 'lora_scale', 10.0),
            },
            "fine_tune_type": "lora",
            "target_modules": getattr(args, 'lora_target_modules', ["self_attn.q_proj", "self_attn.v_proj"]),
            "trainable": True
        }
        print(f"📡 Adapter uploads enabled to server: {args.inference_server_url}")
        
        # Upload initial adapter weights before training starts
        print(f"\n🚀 Uploading initial adapter weights to server...")
        
        async def upload_initial_adapters():
            async with aiohttp.ClientSession() as session:
                server_version = await upload_adapter_weights(
                    session=session,
                    base_url=args.inference_server_url,
                    model=model,
                    adapter_config=adapter_config,
                    expected_policy_version=current_policy_version,
                    step_id=0,
                )
                if server_version != current_policy_version:
                    raise RuntimeError(
                        f"Initial policy version mismatch! Expected {current_policy_version}, "
                        f"but server returned {server_version}. Please restart the server or check configuration."
                    )
        
        # Run the initial upload
        asyncio.run(upload_initial_adapters())
        print(f"✅ Initial weights uploaded. Starting training with policy version {current_policy_version}\n")

    def step(batch):
        nonlocal current_policy_version
        prompt_tokens, targets, prompt_lens, target_lens, type_info = batch

        all_completions, all_completion_texts, all_logprobs, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size,
            inference_server_url=args.inference_server_url,
            step_id=it,  # Pass current iteration as step_id
        )

        mx.clear_cache()

        (lvalue, toks, metrics), grad = loss_value_and_grad(
            model,
            tokenizer=tokenizer,
            batch=(prompt_tokens, targets, prompt_lens, target_lens, type_info),
            completions=all_completions,
            completion_texts=all_completion_texts,
            completion_logprobs=all_logprobs,
            batch_indices=batch_indices,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            ref_model=ref_model,
            grpo_loss_type=args.grpo_loss_type,
            max_tokens=args.max_completion_length,
            importance_sampling_level=args.importance_sampling_level,
        )

        if (it + 1) % args.gradient_accumulation_steps == 0:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            
            # Increment policy version after optimizer update and upload if enabled
            if args.upload_adapters_to_server and adapter_config:
                # Increment local policy version since we've updated the model
                current_policy_version += 1
                print(f"\n🔄 Model updated locally, incrementing to policy version {current_policy_version}")
                print(f"   Uploading updated weights to inference server (iteration {it + 1})...")
                
                async def upload_adapters():
                    nonlocal current_policy_version
                    async with aiohttp.ClientSession() as session:
                        server_version = await upload_adapter_weights(
                            session=session,
                            base_url=args.inference_server_url,
                            model=model,
                            adapter_config=adapter_config,
                            expected_policy_version=current_policy_version,
                            step_id=it + 1,
                        )
                        if server_version != current_policy_version:
                            print(f"⚠️ WARNING: Server returned version {server_version}, expected {current_policy_version}")
                            print(f"   Continuing with local version {current_policy_version}")
                
                # Run the async upload
                asyncio.run(upload_adapters())

        return (lvalue / args.gradient_accumulation_steps), toks, metrics

    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
        'average_generated_tokens': 0,
        "clip_ratio_low": 0,
        "clip_ratio_high": 0,
        "clip_ratio_total": 0,
    }
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0
        accumulated_metrics[f"{func_name}_coverage"] = 0

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ))

        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                temperature=args.temperature,
                iterate_batches=iterate_batches,
                grpo_loss_type=args.grpo_loss_type,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s"
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()

        lvalue, toks, metrics = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            avg_metrics = {}
            for k, v in accumulated_metrics.items():
                val = v / (steps * world_size)
                # Convert to Python scalar if it's an MLX array
                avg_metrics[k] = val.item() if hasattr(val, 'item') else val
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                pbar.set_postfix({
                    'loss': f"{train_loss:.3f}",
                    'it/s': f"{it_sec:.3f}",
                })
                tqdm.write(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"total_r_mean {avg_metrics['total_rewards_mean']:.3f}, "
                    f"total_r_std {avg_metrics['total_rewards_std']:.3f}, "
                    f"group_r_mean {avg_metrics['grouped_rewards_mean']:.3f}, "
                    f"group_r_std {avg_metrics['grouped_rewards_std']:.3f}, "
                    f"kl {avg_metrics['kl']:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"\n"
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")
