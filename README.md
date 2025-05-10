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

### SFT-Training

### ORPO-Training

Odds Ratio Preference Optimization (ORPO) training fine-tunes models using human preference data. Usage:

```shell
mlx_lm_lora.train \
 --model <path_to_model> \
 --train \
 --training-mode orpo \
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

### DPO-Training

### GRPO-Training

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