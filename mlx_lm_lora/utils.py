from pathlib import Path

from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.gguf import convert_to_gguf
from mlx_lm.tuner.utils import dequantize, load_adapters
from .utils import (
    save_model,
    save_config,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.nn as nn


def fuse_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    save_path: str = "fused_model",
    adapter_path: str = "adapters",
    de_quantize: bool = False,
    export_gguf: bool = False,
    gguf_path: str = "ggml-model-f16.gguf",
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