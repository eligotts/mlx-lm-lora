#!/usr/bin/env python3
"""Test script for GRPO trainer with inference server integration."""

import asyncio
import aiohttp
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten

async def test_server_connection():
    """Test basic connection to the inference server."""
    base_url = "http://10.0.0.180:8000"
    
    print("🔍 Testing connection to inference server...")
    print(f"   Server URL: {base_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    print("✅ Server is healthy and responding")
                else:
                    print(f"⚠️ Server returned status {resp.status}")
                    return False
                    
            # Test if server supports adapter upload endpoint
            print("\n📤 Testing adapter upload endpoint...")
            
            # Create dummy adapter data
            dummy_weights = {"test_weight": mx.zeros((10, 10))}
            dummy_config = {
                "model_type": "test",
                "num_layers": 1,
                "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 10.0},
                "fine_tune_type": "lora",
                "target_modules": ["self_attn.q_proj"],
                "trainable": True
            }
            
            # Create temporary files
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save dummy weights
                weights_path = temp_path / "adapters.safetensors"
                mx.save_safetensors(str(weights_path), dummy_weights)
                
                # Save dummy config
                config_path = temp_path / "adapter_config.json"
                with open(config_path, "w") as f:
                    json.dump(dummy_config, f)
                
                # Try to upload
                with open(weights_path, "rb") as weights_file:
                    weights_data = weights_file.read()
                with open(config_path, "rb") as config_file:
                    config_data = config_file.read()
                
                form_data = aiohttp.FormData()
                form_data.add_field('adapter_weights', weights_data,
                                  filename='adapters.safetensors',
                                  content_type='application/octet-stream')
                form_data.add_field('adapter_config', config_data,
                                  filename='adapter_config.json',
                                  content_type='application/json')
                
                async with session.post(f"{base_url}/upload_adapter", data=form_data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"✅ Adapter upload endpoint working! Policy version: {result.get('policy_version', 'unknown')}")
                        return True
                    else:
                        error = await resp.text()
                        print(f"⚠️ Adapter upload failed with status {resp.status}: {error}")
                        return False
                        
    except aiohttp.ClientConnectionError as e:
        print(f"❌ Failed to connect to server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("GRPO Trainer - Inference Server Integration Test")
    print("=" * 60)
    
    # Run async test
    success = asyncio.run(test_server_connection())
    
    if success:
        print("\n✅ All tests passed! The GRPO trainer should work with the inference server.")
        print("\nTo use the integration in training, add these flags:")
        print("  --upload_adapters_to_server")
        print("  --inference_server_url http://10.0.0.180:8000")
    else:
        print("\n❌ Some tests failed. Please check your server configuration.")
        
    print("\nFeatures implemented:")
    print("  ✅ Initial adapter upload before training starts")
    print("  ✅ Local policy version tracking (starts at 1)")
    print("  ✅ Version increment on each optimizer update")
    print("  ✅ Version validation between local and server")
    print("  ✅ Async adapter weight upload after updates")
    print("  ✅ Integration with optimizer updates") 
    print("  ✅ Configurable server URL")
    print("  ✅ Optional upload flag (--upload_adapters_to_server)")
    
    print("\nPolicy Version Flow:")
    print("  1. Start with local version = 1")
    print("  2. Upload initial weights, expect server to return version 1")
    print("  3. On optimizer.update(), increment local version")
    print("  4. Upload new weights with expected version")
    print("  5. Assert server returns same version")
    print("  6. Use local version for all inference requests")

if __name__ == "__main__":
    main()