from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

@register_reward_function("custom_accuracy")
def custom_accuracy_reward(prompts, completions, answers, types=None):
    """
    Custom accuracy reward function that gives 2.0 for correct answers, 0.0 otherwise.
    """
    if not completions or not answers:
        return [0.0] * len(prompts)
    
    # Extract answers from completions (assuming XML format)
    extracted_responses = []
    for completion in completions:
        try:
            # Extract answer from <answer> tags
            if "<answer>" in completion and "</answer>" in completion:
                answer_start = completion.find("<answer>") + 8
                answer_end = completion.find("</answer>")
                extracted_answer = completion[answer_start:answer_end].strip()
                extracted_responses.append(extracted_answer)
            else:
                extracted_responses.append("")
        except:
            extracted_responses.append("")
    
    # Compare with ground truth answers
    rewards = []
    for extracted, ground_truth in zip(extracted_responses, answers):
        if extracted and ground_truth and extracted.strip() == ground_truth.strip():
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    
    return rewards

@register_reward_function("length_penalty")
def length_penalty_reward(prompts, completions, answers, types=None):
    """
    Reward function that penalizes very short or very long responses.
    """
    if not completions:
        return [0.0] * len(prompts)
    
    rewards = []
    for completion in completions:
        length = len(completion)
        # Penalize very short (< 10 chars) or very long (> 500 chars) responses
        if length < 10:
            rewards.append(-0.5)
        elif length > 500:
            rewards.append(-0.3)
        else:
            rewards.append(0.1)  # Small positive reward for reasonable length
    
    return rewards

@register_reward_function("format_check")
def format_check_reward(prompts, completions, answers, types=None):
    """
    Reward function that checks for proper XML formatting.
    """
    if not completions:
        return [0.0] * len(prompts)
    
    rewards = []
    for completion in completions:
        # Check for proper XML structure
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        
        if has_think and has_answer:
            # Check if tags are in correct order
            think_start = completion.find("<think>")
            think_end = completion.find("</think>")
            answer_start = completion.find("<answer>")
            answer_end = completion.find("</answer>")
            
            if think_start < think_end < answer_start < answer_end:
                rewards.append(0.5)  # Perfect format
            else:
                rewards.append(0.2)  # Has tags but wrong order
        elif has_think or has_answer:
            rewards.append(0.1)  # Partial format
        else:
            rewards.append(0.0)  # No format
    
    return rewards

@register_reward_function("content_quality")
def content_quality_reward(prompts, completions, answers, types=None):
    """
    Reward function that checks for content quality indicators.
    """
    if not completions:
        return [0.0] * len(prompts)
    
    rewards = []
    for completion in completions:
        score = 0.0
        
        # Check for reasoning indicators
        reasoning_indicators = ["because", "since", "therefore", "thus", "hence", "as a result"]
        for indicator in reasoning_indicators:
            if indicator.lower() in completion.lower():
                score += 0.1
        
        # Check for mathematical expressions
        if any(char in completion for char in ["+", "-", "*", "/", "="]):
            score += 0.2
        
        # Check for step-by-step reasoning
        if "step" in completion.lower() or "first" in completion.lower() and "then" in completion.lower():
            score += 0.3
        
        # Cap the reward
        rewards.append(min(score, 1.0))
    
    return rewards 