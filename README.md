# mlx-lm-lora
Train Large Language Models localy on Apple Silicon using MLX. Fine-tuning works with all the model that are supported with MLX-LM, for example:

- Llama, 3, 4
- Phi2, 3
- Mistral
- Mixtral
- Qwen2, 2.5, 3
- Qwen3 MoE
- Gemma1, 2, 3
- OLMo, OLMoE
- MiniCPM, MiniCPM3
- and more ...

## Contents

- [Run](#Run)
  - [LoRA or Full-Precision](#Lora-or-Full-Precision)
  - [SFT](#SFT-Training)
  - [ORPO-Training](#ORPO-Training)
  - [DPO-Training](#DPO-Training)
  - [GRPO-Training](#GRPO-Training)
  - [Evaluate](#Evaluate)
  - [Generate](#Generate)
- [Fuse](#Fuse)
- [Memory Issues](#Memory-Issues)

--

## Run

The main command is `mlx_lm_lora.train`. To see a full list of command-line options run:

```shell
mlx_lm_lora.train --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the
[example YAML](examples/lora_config.yaml). For example:

```shell
mlx_lm_lora.train --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding
values in the config.

---

### LoRA or Full-Precision

To fine-tune a model use:

```shell
mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 600
```

To fine-tune the full model weights, add the `--train-type full` flag.
Currently supported training types are `lora` (default), `dora`, and `full`.

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl`
when using `--train` and a path to a `test.jsonl` when using `--test`. For more

If `--model` points to a quantized model, then the training will use QLoRA,
otherwise it will use regular LoRA.

By default, the adapter config and learned weights are saved in `adapters/`.
You can specify the output location with `--adapter-path`.

You can resume fine-tuning with an existing adapter with
`--resume-adapter-file <path_to_adapters.safetensors>`.

---

### SFT-Training

Supervised Fine-Tuning (SFT) trains a model using pairs of prompts and expected completions. This is the most common form of instruction tuning.

To run SFT:

mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data>

You can set the training type explicitly using --train-type lora, dora, or full. By default, LoRA is used.

Data Format

The data should be in JSONL format with one of the following structures:

Chat-style (preferred for chat models):

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}

Prompt-completion style:

{"prompt": "What is the capital of France?", "completion": "Paris."}

You can mix system messages or use multi-turn chat formatting depending on the target model’s capabilities.

Additional Options
	•	--use-chat-template: Format messages using the model’s chat template (default: False)
	•	--mask-prompt: Apply loss only on the assistant’s output instead of the full sequence
	•	--train-type: Choose lora, dora, or full for LoRA, DoRA, or full-weight fine-tuning
	•	--resume-adapter-file: Resume training from a previously saved adapter

Example with chat template and prompt masking:

mlx_lm_lora.train \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --use-chat-template True \
    --mask-prompt

---

### ORPO-Training

Odds Ratio Preference Optimization (ORPO) training fine-tunes models using human preference data. Usage:

```shell
mlx_lm_lora.train \
 --model <path_to_model> \
 --train \
 --train-mode orpo \
 --data <path_to_data> \
 --beta 0.1
```

Parameters:

- `--beta`: Temperature for logistic function (default: 0.1)

Data format (JSONL):

```jsonl
# Basic format with string responses
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}

# With custom preference score
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response", "preference_score": 8.0}

# With system message
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response", "system": "System instruction"}

# With full conversation objects
{
  "prompt": "User prompt",
  "chosen": {
    "messages": [
      {"role": "system", "content": "System instruction"},
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Assistant response"}
    ]
  },
  "rejected": {
    "messages": [
      {"role": "system", "content": "System instruction"},
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Assistant response"}
    ]
  }
}
```

The trainer assigns binary rewards (1.0 chosen, 0.0 rejected) if no explicit rewards provided via `preference_score`.

---

### DPO-Training

Direct Preference Optimization (DPO) training allows you to fine-tune models using human preference data. To use DPO training, set the training mode to 'dpo':

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --train-mode dpo \
    --data <path_to_data> \
    --beta 0.1
```

The DPO training accepts the following additional parameters:

- `--beta`: Controls the strength of the DPO loss (default: 0.1)
- `--dpo-loss-type`: Choose between "sigmoid" (default), "hinge", "ipo", or "dpop" loss functions
- `--delta`: Margin parameter for hinge loss (default: 50.0)
- `--reference-model-path`: Path to a reference model for DPO training

For DPO training, the data should be in JSONL format with the following structure:

```jsonl
{"prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}
```

if the Prompt template accept a system message, you can extend the Dataset with a additional "system" field.

```jsonl
{"system": "You are a helpfull assistant", "prompt": "User prompt", "chosen": "Preferred response", "rejected": "Less preferred response"}
```

---

### GRPO-Training
#### Overview

Group Relative Policy Optimization (GRPO) is a fine-tuning method that optimizes language models by generating multiple responses per prompt and learning from their relative quality. This approach helps improve response quality through comparative learning.

#### Dataset Format

GRPO requires a dataset in JSONL format (one JSON object per line) with the following structure:

```json
{"prompt": "Your question or instruction here", "answer": "The expected response"}
```

Each entry must contain:
- `prompt`: The input text for the model to respond to
- `answer`: The target/reference response

Optional fields:
- `system`: A system message providing context or instructions for the model

Example entries:
```json
{"prompt": "Gerald spends $100 a month on baseball supplies. His season is 4 months long. He wants to use the months he's not playing baseball to save up by raking, shoveling, and mowing lawns. He charges $10 for each. How many chores does he need to average a month to save up for his supplies?", "answer": "5"}
{"prompt": "Ann is cutting fabric to make curtains. She cuts a 4 foot by 6 foot rectangle for the living room, and a 2 foot by 4 foot rectangle for the bedroom. If the bolt of fabric is 16 feet by 12 feet, how much fabric is left in square feet?", "answer": "160"}
{"prompt": "Arnel had ten boxes of pencils with the same number of pencils in each box. He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?", "answer": "5", "system": "You are a helpful math tutor."}
```

#### Usage

To fine-tune a model using GRPO:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --fine-tune-type grpo \
    --group-size 4
```

#### GRPO-Specific Arguments

- `--group-size`: Number of responses generated per prompt (default: 4)
- `--beta`: KL penalty coefficient for policy optimization (default: 0.1)
- `--epsilon`: Small constant for numerical stability (default: 1e-4)
- `--max-completion-length`: Maximum length of generated completions (default: 512)
- `--reference-model-path`: Path to reference model weights. If not specified, uses the same model
- `--temperature`: Sampling temperature for generations. Higher values increase randomness (default: 1.0)
- `--reward-weights`: Optional list of weights for multiple reward functions. Must match number of reward functions. If not specified, all rewards weighted equally with 1.0
- `--use-chat-template`: Whether to use the model's chat template for formatting prompts (default: False)
- `--use-prompt`: Whether to use the prompt as part of the input for generation (default: False)

#### Training Process

During GRPO training, the model:
1. Takes each prompt from the dataset
2. Generates multiple responses (specified by `--group-size`)
3. Evaluates these responses against the reference answer
4. Optimizes the policy based on the relative quality of the responses

#### Resource Considerations

GRPO requires more compute resources than standard LoRA training since it generates multiple responses per prompt. Consider:
- Reducing batch size
- Using gradient checkpointing
- Adjusting `--group-size` to balance between quality and resource usage

If running into memory issues, you can also try:
- Reducing `--max-completion-length`
- Using a smaller model for initial experiments

#### Example Command with Full Options

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --fine-tune-type grpo \
    --group-size 4 \
    --beta 0.1 \
    --epsilon 1e-4 \
    --max-completion-length 512 \
    --reference-model-path <optional_path_to_reference_model> \
    --temperature 1.0 \
    --reward-weights 1.0 1.0 \
    --use-chat-template False \
    --use-prompt False \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --num-epochs 3
```

---

### Evaluate

To compute test set perplexity use:

```shell
mlx_lm_lora.train \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx-lm` with `mlx_lm.generate`:

```shell
mlx_lm.generate \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --prompt "<your_model_prompt>"
```

#### Prompt Masking

The default training computes a loss for every token in the sample. You can
ignore the prompt and compute loss for just the completion by passing
`--mask-prompt`. Note this is only supported for `chat` and `completion`
datasets. For `chat` datasets the final message in the message list is
considered the completion.

---

## Fuse

You can generate a model fused with the low-rank adapters using the
`mlx_lm_lora.fuse` command. This command also allows you to optionally:

- Upload the fused model to the Hugging Face Hub.
- Export the fused model to GGUF. Note GGUF support is limited to Mistral,
  Mixtral, and Llama style models in fp16 precision.

To see supported options run:

```shell
mlx_lm_lora.fuse --help
```

To generate the fused model run:

```shell
mlx_lm_lora.fuse --model <path_to_model>
```

This will by default load the adapters from `adapters/`, and save the fused
model in the path `fused_model/`. All of these are configurable.

To upload a fused model, supply the `--upload-repo` and `--hf-path` arguments
to `mlx_lm_lora.fuse`. The latter is the repo name of the original model, which is
useful for the sake of attribution and model versioning.

For example, to fuse and upload a model derived from Mistral-7B-v0.1, run:

```shell
mlx_lm_lora.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --upload-repo mlx-community/my-lora-mistral-7b \
    --hf-path mistralai/Mistral-7B-v0.1
```

To export a fused model to GGUF, run:

```shell
mlx_lm_lora.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --export-gguf
```

This will save the GGUF model in `fused_model/ggml-model-f16.gguf`. You
can specify the file name with `--gguf-path`.

---

## Memory Issues

Fine-tuning a large model with LoRA requires a machine with a decent amount
of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization (QLoRA). You can use QLoRA by generating a quantized model
   with `convert.py` and the `-q` flag. See the [Setup](#setup) section for
   more details.

2. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

3. Reduce the number of layers to fine-tune with `--num-layers`. The default
   is `16`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

5. Gradient checkpointing lets you trade-off memory use (less) for computation
   (more) by recomputing instead of storing intermediate values needed by the
   backward pass. You can use gradient checkpointing by passing the
   `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for
   larger batch sizes or sequence lengths with smaller or quantized models.