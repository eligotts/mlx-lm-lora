from mlx_lm.generate import generate
import mlx.nn as nn
import mlx as mx

from typing import Optional


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''

DEFAULT_PAIRWISE_HUMAN_PROMPT = '''## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Reply with the identifier (0, 1) of the best model.
'''


class LLMPairwiseJudge():
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        if shuffle_order:
            flip_mask = mx.random.randint(0, 2, (len(prompts),)).astype(bool)
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": content}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=16
            )
            if response in ["0", "1"]:
                return int(response)
            else:
                print(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        ranks = []
        for prompt, completion in zip(prompts, completions):
            ranks.append(get_rank(prompt, completion))

        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        return ranks
    

class HumanPairwiseJudge():
    def __init__(self, prompt: Optional[str] = None,):
        self.prompt = prompt or DEFAULT_PAIRWISE_HUMAN_PROMPT

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        if shuffle_order:
            flip_mask = mx.random.randint(0, 2, (len(prompts),)).astype(bool)
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        def get_rank(prompt, candidates):
            content = self.prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            response = input(f"Choose with one is better:\n{content}")
            if response in ["0", "1"]:
                return int(response)
            else:
                print(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        ranks = []
        for prompt, completion in zip(prompts, completions):
            ranks.append(get_rank(prompt, completion))

        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        return ranks