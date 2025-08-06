"""Example implementation of InferenceClient for vLLM-like inference servers."""

import aiohttp
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
import json
import time
import mlx.core as mx
from pathlib import Path
from mlx.utils import tree_flatten

from .async_batch_generator import InferenceClient


class MLXInferenceClient(InferenceClient):
    """MLX inference client for the MLX RL inference server.
    
    This client communicates with the MLX inference server that supports:
    - LoRA adapter uploads via multipart form data
    - Batch generation with logprobs
    - Policy version tracking
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__(host, port)
        self.base_url = f"http://{host}:{port}"
        self.session = None
        self.current_step_id = 0
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def generate_batch(
        self,
        prompt_tokens: List[List[int]],
        prompt_texts: List[str],
        tokenizer,
        max_tokens: int = 512,
        group_size: int = 4,
        temperature: float = 0.8,
        batch_size: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completions for a batch of prompts.
        
        Args:
            prompt_tokens: List of tokenized prompts
            prompt_texts: List of prompt strings  
            tokenizer: Tokenizer for encoding/decoding
            max_tokens: Maximum tokens to generate
            group_size: Number of completions per prompt
            temperature: Sampling temperature
            batch_size: Batch size for generation
            
        Returns:
            Dictionary with completions, completion_texts, and batch_indices
        """
        print(f"\n[MLXInferenceClient] Starting batch generation:")
        print(f"  - Prompts: {len(prompt_texts)}")
        print(f"  - Group size: {group_size}")
        print(f"  - Total requests: {len(prompt_texts) * group_size}")
        print(f"  - Policy version: {self._weight_version}")
        print(f"  - Step ID: {self.current_step_id}")
        
        await self._ensure_session()
        
        all_completions = []
        all_completion_texts = []
        batch_indices = []
        
        # Format prompts with chat template
        print(f"[MLXInferenceClient] Formatting prompts with chat template...")
        prompts_formatted = [[{"role": "user", "content": prompt}] for prompt in prompt_texts]
        prompts_formatted = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) 
                           for prompt in prompts_formatted]
        
        # Generate requests for all prompt/group combinations
        generation_requests = []
        for i, prompt in enumerate(prompts_formatted):
            for k in range(group_size):
                request_data = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "logprobs": 1,  # Request logprobs for GRPO
                    "policy_version": self._weight_version,
                    "step_id": self.current_step_id,
                    "request_id": f"batch-{i}-{k}"
                }
                generation_requests.append((i, request_data))
        
        # Submit all requests concurrently - server will batch them
        print(f"[MLXInferenceClient] Submitting {len(generation_requests)} requests to server...")
        tasks = []
        for idx, request_data in generation_requests:
            task = self.session.post(f"{self.base_url}/generate", json=request_data)
            tasks.append((idx, task))
        
        # Collect all responses
        print(f"[MLXInferenceClient] Waiting for responses...")
        responses = []
        success_count = 0
        error_count = 0
        
        for idx, task in tasks:
            try:
                async with task as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[MLXInferenceClient] ERROR: Request {idx} failed - Status {response.status}, {error_text}")
                        result = None
                        error_count += 1
                    else:
                        result = await response.json()
                        success_count += 1
                responses.append((idx, result))
            except Exception as e:
                print(f"[MLXInferenceClient] ERROR: Request {idx} exception - {e}")
                responses.append((idx, None))
                error_count += 1
        
        print(f"[MLXInferenceClient] Received {success_count} successful responses, {error_count} errors")
        
        # Process responses
        for idx, result in responses:
            if result is not None:
                # Extract completion text and tokens
                completion_text = result["text"]
                completion_ids = mx.array(result["completion_ids"])
                
                all_completions.append(mx.stop_gradient(completion_ids))
                all_completion_texts.append(completion_text)
                batch_indices.append(idx)
            else:
                # Add empty completion on error
                all_completions.append(mx.array([]))
                all_completion_texts.append("")
                batch_indices.append(idx)
        
        self.current_step_id += 1
        
        print(f"[MLXInferenceClient] Batch generation complete:")
        print(f"  - Completions: {len(all_completions)}")
        print(f"  - Average length: {sum(len(c) for c in all_completion_texts) / len(all_completion_texts) if all_completion_texts else 0:.1f} chars")
        
        return {
            "completions": all_completions,
            "completion_texts": all_completion_texts,
            "batch_indices": batch_indices,
        }
    
    def update_weights(self, model_state: Dict[str, mx.array], adapter_config: Optional[Dict[str, Any]] = None):
        """Update LoRA adapter weights on inference server.
        
        Args:
            model_state: Model state dictionary containing LoRA adapter weights
            adapter_config: Optional adapter configuration
        """
        # Create async task to send weights
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _upload_adapter_async():
            print(f"[MLXInferenceClient] Preparing adapter upload")
            await self._ensure_session()
            
            # Save adapter weights to temporary file
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save adapter weights
                weights_path = temp_path / "adapters.safetensors"
                print(f"[MLXInferenceClient] Saving {len(model_state)} parameters to temporary file")
                mx.save_safetensors(str(weights_path), model_state)
                
                # Get file size
                import os
                file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
                print(f"[MLXInferenceClient] Adapter file size: {file_size:.2f} MB")
                
                # Create adapter config if not provided
                if adapter_config is None:
                    print(f"[MLXInferenceClient] No adapter config provided, using defaults")
                    # Extract LoRA config from model state keys
                    # This is a simplified version - adjust based on your model structure
                    adapter_config = {
                        "model_type": "unknown",  # Should be provided by caller
                        "num_layers": -1,  # Should be provided by caller
                        "lora_parameters": {
                            "rank": 16,
                            "scale": 10.0,
                            "dropout": 0.0
                        },
                        "fine_tune_type": "lora",
                        "target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
                        "trainable": True
                    }
                else:
                    print(f"[MLXInferenceClient] Using provided adapter config")
                
                config_path = temp_path / "adapter_config.json"
                with open(config_path, "w") as f:
                    json.dump(adapter_config, f, indent=2)
                
                # Create multipart form data
                print(f"[MLXInferenceClient] Creating multipart form data")
                form_data = aiohttp.FormData()
                
                with open(weights_path, "rb") as f:
                    form_data.add_field('adapter_weights', f.read(),
                                      filename='adapters.safetensors',
                                      content_type='application/octet-stream')
                
                with open(config_path, "rb") as f:
                    form_data.add_field('adapter_config', f.read(),
                                      filename='adapter_config.json',
                                      content_type='application/json')
                
                # Upload adapter
                print(f"[MLXInferenceClient] Uploading adapter to {self.base_url}/upload_adapter")
                upload_start = time.time()
                
                async with self.session.post(
                    f"{self.base_url}/upload_adapter",
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as response:
                    upload_time = time.time() - upload_start
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[MLXInferenceClient] ERROR: Upload failed after {upload_time:.2f}s")
                        raise RuntimeError(f"Failed to upload adapter: Status {response.status}, {error_text}")
                    
                    result = await response.json()
                    new_version = result.get("policy_version", self._weight_version + 1)
                    print(f"[MLXInferenceClient] Adapter uploaded successfully in {upload_time:.2f}s")
                    print(f"[MLXInferenceClient] New policy version: {new_version}")
                    
        loop.run_until_complete(_upload_adapter_async())
        loop.close()
        
        self._weight_version += 1
        
    def __del__(self):
        """Cleanup session on deletion."""
        if self.session:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.session.close())
            loop.close()


class MockInferenceClient(InferenceClient):
    """Mock inference client for testing without a real server.
    
    This client simulates async generation using the local model.
    Useful for testing the async infrastructure.
    """
    
    def __init__(self, model, host: str = "localhost", port: int = 8080):
        super().__init__(host, port)
        self.model = model
        self.is_generating = False
        
    async def generate_batch(
        self,
        prompt_tokens: List[List[int]],
        tokenizer,
        max_tokens: int = 512,
        group_size: int = 4,
        temperature: float = 0.8,
        batch_size: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completions using local model (for testing)."""
        self.is_generating = True
        
        # Import the generate_grpo function
        from .grpo_trainer import generate_grpo
        
        # Simulate async delay
        await asyncio.sleep(0.1)
        
        # Use local model for generation
        completions, completion_texts, batch_indices = generate_grpo(
            model=self.model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size,
        )
        
        self.is_generating = False
        
        return {
            "completions": completions,
            "completion_texts": completion_texts,
            "batch_indices": batch_indices,
        }
        
    def _send_weights_to_server(self, weights: Dict[str, np.ndarray]):
        """Mock weight update (no-op for testing)."""
        print(f"Mock: Updated weights to version {self._weight_version + 1}")
        # In a real implementation, this would send weights to the server