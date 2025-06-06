from dataclasses import dataclass, field
from typing import List, Union
from pathlib import Path
import time

from transformers import PreTrainedTokenizer

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.generate import generate
from mlx_lm.utils import load

from .sft_trainer import SFTTrainingArgs, grad_checkpoint
from .judge import LLMPairwiseJudge, HumanPairwiseJudge
from .dpo_trainer import get_token_scores

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class OnlineDPOTrainingArgs(SFTTrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for DPO training."}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'."},
    )
    delta: float = field(
        default=50.0, metadata={"help": "Delta parameter for DPOP loss type."}
    )
    judge: str = field(
        default=None,
        metadata={"help": "What LLM to use as the judge, if left empty, it#s going to be you (human)."}
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


def generate_for_online_dpo(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: list,
    max_tokens: int = 512
) -> list[list[str]]:
    completions = []
    for prompt in prompts:
        prompt_input_ids = prompt
        generated_1 = generate(model, tokenizer, prompt_input_ids, max_tokens=max_tokens)
        generated_2 = generate(model, tokenizer, prompt_input_ids, max_tokens=max_tokens)
        completions.append([generated_1, generated_2])
    return completions


def compute_score(scores, mask, loss_type):
    token_count = mask.sum(-1)
    return scores.sum(-1) / token_count if loss_type == "ipo" else scores.sum(-1)


def dpo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    reference_chosen_score: mx.array,
    reference_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
):
    # Preference logits
    logits = (policy_chosen_score - policy_rejected_score) - \
             (reference_chosen_score - reference_rejected_score)

    # Loss calculation
    if loss_type == "sigmoid":
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen_score),
            reference_chosen_score - policy_chosen_score,
        )
        losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Token counts and rewards
    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score / num_rejected_tokens),
        "policy_chosen_logps": mx.mean(policy_chosen_score / num_chosen_tokens),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
    }

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def iterate_online_dpo_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["prompt"]))

    step = mx.distributed.init().size()

    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")
    
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = (
            np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        )

        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            prompt = [len(x["prompt"]) for x in batch]

            yield prompt
        if not train:
            break


def evaluate_online_dpo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    loss_fn: callable = dpo_loss,
    judge: str = None
):
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_online_dpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        prompts = batch
        

        policy_chosen_scores = get_token_scores(model, chosen, chosen_masks)
        policy_rejected_scores = get_token_scores(model, rejected, rejected_masks)

        policy_chosen_score = compute_score(policy_chosen_scores, chosen_masks, loss_type)
        policy_rejected_score = compute_score(policy_rejected_scores, rejected_masks, loss_type)

        if ref_model is None:
            reference_chosen_logprobs = mx.zeros_like(policy_chosen_score)
            reference_rejected_logprobs = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_logprobs = mx.stop_gradient(get_token_scores(ref_model, chosen, chosen_masks))
            ref_rejected_logprobs = mx.stop_gradient(get_token_scores(ref_model, rejected, rejected_masks))
            reference_chosen_logprobs = compute_score(ref_chosen_logprobs, chosen_masks, loss_type)
            reference_rejected_logprobs = compute_score(ref_rejected_logprobs, rejected_masks, loss_type)

        loss_value, reward, toks, metrics = loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            loss_type=loss_type,
            beta=beta,
            delta=delta,
        )
        all_losses += loss_value * toks
        all_rewards += reward
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, all_rewards, ntokens)
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_online_dpo(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: OnlineDPOTrainingArgs = OnlineDPOTrainingArgs(),
    loss_fn: callable = dpo_loss,
    training_callback: TrainingCallback = None,
    loss_type="sigmoid",
    judge="mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit",
    max_tokens: int = 512
):
    print(f"Starting Online DPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        prompts, mask = batch
        
        # Generate completions for each prompt
        completions = generate_for_online_dpo(model, tokenizer, prompts, max_tokens=max_tokens)
        
        # Judge the completions
        if judge is None:
            judger = HumanPairwiseJudge()
            judged = judger.judge(prompts, completions=completions)  # returns list[int]
        else:
            judge_model, judge_tokenizer = load(judge)
            judger = LLMPairwiseJudge(model=judge_model, tokenizer=judge_tokenizer)
            judged = judger.judge(prompts, completions=completions)  # returns list[int]
        
        # Process judged results to create chosen/rejected pairs
        chosen = []
        rejected = []
        for i, (prompt, completion_pair, judgment) in enumerate(zip(prompts, completions, judged)):
            if judgment == 0:  # First completion is preferred
                chosen.append(prompt + completion_pair[0])
                rejected.append(prompt + completion_pair[1])
            else:  # Second completion is preferred
                chosen.append(prompt + completion_pair[1])
                rejected.append(prompt + completion_pair[0])
        
        # Tokenize chosen and rejected
        chosen_tokens = [mx.array(tokenizer.encode(text)) for text in chosen]
        rejected_tokens = [mx.array(tokenizer.encode(text)) for text in rejected]
        
        # Create masks
        chosen_masks = [mx.ones(len(tokens)) for tokens in chosen_tokens]
        rejected_masks = [mx.ones(len(tokens)) for tokens in rejected_tokens]
        
        # Get policy scores
        policy_chosen_logprobs = [get_token_scores(model, tokens.unsqueeze(0), mask.unsqueeze(0)) 
                                 for tokens, mask in zip(chosen_tokens, chosen_masks)]
        policy_rejected_logprobs = [get_token_scores(model, tokens.unsqueeze(0), mask.unsqueeze(0)) 
                                   for tokens, mask in zip(rejected_tokens, rejected_masks)]
        
        policy_chosen_score = [compute_score(logprobs, mask, loss_type) 
                              for logprobs, mask in zip(policy_chosen_logprobs, chosen_masks)]
        policy_rejected_score = [compute_score(logprobs, mask, loss_type) 
                                for logprobs, mask in zip(policy_rejected_logprobs, rejected_masks)]
        
        # Get reference scores
        if ref_model is None:
            reference_chosen_logprobs = mx.zeros_like(policy_chosen_score)
            reference_rejected_logprobs = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_logprobs = [mx.stop_gradient(get_token_scores(ref_model, tokens.unsqueeze(0), mask.unsqueeze(0)))
                                  for tokens, mask in zip(chosen_tokens, chosen_masks)]
            ref_rejected_logprobs = [mx.stop_gradient(get_token_scores(ref_model, tokens.unsqueeze(0), mask.unsqueeze(0)))
                                    for tokens, mask in zip(rejected_tokens, rejected_masks)]
            reference_chosen_logprobs = [compute_score(logprobs, mask, loss_type) 
                                        for logprobs, mask in zip(ref_chosen_logprobs, chosen_masks)]
            reference_rejected_logprobs = [compute_score(logprobs, mask, loss_type) 
                                          for logprobs, mask in zip(ref_rejected_logprobs, rejected_masks)]
        
        # Convert to arrays
        policy_chosen_score = mx.array(policy_chosen_score)
        policy_rejected_score = mx.array(policy_rejected_score)
        reference_chosen_score = mx.array(reference_chosen_logprobs)
        reference_rejected_score = mx.array(reference_rejected_logprobs)
        chosen_masks = mx.array([mask.sum() for mask in chosen_masks])
        rejected_masks = mx.array([mask.sum() for mask in rejected_masks])
        
        # Compute loss and gradients
        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            policy_chosen_score, policy_rejected_score, 
            reference_chosen_score, reference_rejected_score, 
            chosen_masks=chosen_masks, rejected_masks=rejected_masks
        )
        
        if (it + 1) % args.gradient_accumulation_steps == 0:
            grad = average_gradients(grad)
            optimizer.update(model, grad)

        return (lvalue / args.gradient_accumulation_steps), toks, metrics

    def loss_wrapper(policy_chosen_score, policy_rejected_score, reference_chosen_score, reference_rejected_score, chosen_masks, rejected_masks):
        return loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=loss_type,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "accuracies": 0,
        "margins": 0,
        "policy_rejected_logps": 0,
        "policy_chosen_logps": 0,
        "rejected_logits_mean": 0,
        "chosen_logits_mean": 0,
    }

    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_dpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_dpo(
                model=model,
                ref_model=ref_model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
                loss_type=loss_type,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val accuracy {val_metrics['accuracies']:.3f}, "
                    f"Val margin {val_metrics['margins']:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        "val_chosen_reward": val_rewards[0],
                        "val_rejected_reward": val_rewards[1],
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            start = time.perf_counter()

        lvalue, reward, toks, metrics = step(batch)
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = mx.distributed.all_sum(rewards).tolist()
            train_rewards = [r / (steps * world_size) for r in train_rewards]
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Chosen reward {train_rewards[0]:.3f}, "
                    f"Rejected reward {train_rewards[1]:.3f}, "
                    f"Accuracy {avg_metrics['accuracies']:.3f}, "
                    f"Margin {avg_metrics['margins']:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
