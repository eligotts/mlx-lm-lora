from typing import Optional
from pathlib import Path

from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.gguf import convert_to_gguf
from mlx_lm.tuner.utils import dequantize, load_adapters, print_trainable_parameters, linear_to_lora_layers
from mlx_lm.utils import (
    save_model,
    save_config,
    load,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.nn as nn


def fuse_and_save_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    save_path: str = "fused_model",
    adapter_path: Optional[str] = None,
    de_quantize: Optional[bool] = False,
    export_gguf: Optional[bool] = False,
    gguf_path: Optional[str] = "ggml-model-f16.gguf",
) -> None:
    """
    Fuse fine-tuned adapters into the base model.
    
    Args:
        model: The MLX model to fuse adapters into.
        tokenizer: The tokenizer wrapper.
        save_path: The path to save the fused model.
        adapter_path: Path to the trained adapter weights and config.
        de_quantize: Generate a de-quantized model.
        export_gguf: Export model weights in GGUF format.
        gguf_path: Path to save the exported GGUF format model weights.
    """
    model.freeze()

    if adapter_path is not None:
        print(f"Loading adapters from {adapter_path}")
        model = load_adapters(model, adapter_path)

    args = model.args

    fused_linears = [
        (n, m.fuse(de_quantize=de_quantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        print("De-quantizing model")
        model = dequantize(model)  # Fixed: was model_obj, should be model
        args.pop("quantization", None)

    save_path_obj = Path(save_path)
    save_model(save_path_obj, model, donate_model=True)
    save_config(args, config_path=save_path_obj / "config.json")
    tokenizer.save_pretrained(save_path_obj)

    if export_gguf:
        model_type = args["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        weights = dict(tree_flatten(model.parameters()))
        convert_to_gguf(save_path, weights, args, str(save_path_obj / gguf_path))


def from_pretrained(
    model: str,
    adapter_path: Optional[str] = None,
    lora_config: Optional[dict] = None,
    quantized_load: Optional[dict] = None,
) -> nn.Module:
    """
    Load a model with LoRA adapters and optional quantization.
    Args:
        model: The base MLX model to load.
        lora_config: Configuration for LoRA adapters.
        quantized_load: If provided, the model will be loaded with quantization.
    Returns:
        nn.Module: The model with LoRA adapters loaded.
        tokenizer: The tokenizer associated with the model.
    """

    print(f"Loading model {model}")
    model, tokenizer = load(model, adapter_path=adapter_path)

    if lora_config is not None:
        print(f"Loading LoRA adapters with config: {lora_config}")
        this_lora_config = {
            "rank": getattr(lora_config, "rank", 8),
            "dropout": getattr(lora_config, "dropout", 0.0),
            "scale": getattr(lora_config, "scale", 10.0),
            "use_dora": getattr(lora_config, "use_dora", False),
        }

        model.freeze()
        linear_to_lora_layers(
            model=model,
            num_layers=getattr(lora_config, "num_layers", None),
            config=this_lora_config,
            use_dora=getattr(lora_config, "use_dora", False),
        )
    
    if quantized_load is not None:
        print(f"Quantizing model with {quantized_load} bits")
        nn.quantize(
            model,
            bits=getattr(quantized_load, "bits", 4),
            group_size=getattr(quantized_load, "group_size", 128),
        )

    return model, tokenizer