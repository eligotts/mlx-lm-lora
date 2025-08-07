# GRPO Trainer - Inference Server Integration

## Overview
This implementation adds real-time LoRA adapter weight synchronization between the GRPO trainer and an inference server. The system maintains strict policy version tracking to ensure consistency between training and inference.

## Key Features

### 1. Policy Version Tracking
- **Local version tracking**: Starts at version 1 and increments with each optimizer update
- **Server validation**: Asserts that server policy versions match local expectations
- **Consistent state**: Ensures training and inference use the same model weights

### 2. Weight Upload Flow
1. **Initial upload**: Before training starts, uploads initial LoRA weights (version 1)
2. **Training updates**: After each `optimizer.update()`:
   - Increment local policy version
   - Upload new weights to server
   - Validate server returns matching version
3. **Generation requests**: All inference requests include the current policy version

### 3. Configuration
Add these parameters to `GRPOTrainingArgs`:
```python
args = GRPOTrainingArgs(
    # ... other args ...
    upload_adapters_to_server=True,  # Enable server uploads
    inference_server_url="http://10.0.0.180:8000",  # Server URL
)
```

## Implementation Details

### Files Modified
- `mlx_lm_lora/trainer/grpo_trainer.py`: Core implementation
- `examples/grpo_2.py`: Added usage documentation

### New Functions
- `upload_adapter_weights()`: Async function to upload weights and validate versions
- Modified `generate_grpo()`: Uses policy version for inference requests
- Modified `train_grpo()`: Handles initial upload and version tracking

### Version Consistency Protocol
```
Training Side                    Server Side
-------------                    -----------
Start: version = 1               
Upload initial weights --------> Expects version 1
                                 Returns: version 1
Assert match ✓                   

[Training iteration]
optimizer.update()
version = 2
Upload new weights ------------> Expects version 2
                                 Returns: version 2
Assert match ✓

Generate request ---------------> Server uses its current version
                                 Returns results with version info
```

Note: The inference server manages policy versions internally and doesn't accept version parameters in generation requests. It always uses the most recently uploaded weights.

## Error Handling
- **Version mismatch**: Warns but continues with local version
- **Upload failure**: Raises exception to prevent inconsistent state
- **Server unavailable**: Training fails early with clear error message

## Testing
Run the test script to verify server connectivity:
```bash
python test_grpo_server_integration.py
```

## Benefits
1. **Real-time updates**: Inference server always has latest model weights
2. **Version consistency**: Guarantees training and inference use same weights
3. **Clear tracking**: Easy to debug which model version is being used
4. **Optional feature**: Can be disabled for local-only training