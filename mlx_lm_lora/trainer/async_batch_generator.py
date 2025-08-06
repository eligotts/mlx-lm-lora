"""Async batch generation infrastructure for train-one-off methodology."""

import asyncio
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten


@dataclass
class BatchRequest:
    """Request for batch generation."""
    batch_id: int
    env_inputs: Dict[str, Any]
    max_concurrent: Optional[int] = None
    generation_timeout: float = 600.0


@dataclass 
class BatchResult:
    """Result from batch generation."""
    batch_id: int
    processed_results: Dict[str, Any]
    generation_time: float
    error: Optional[str] = None


class AsyncBatchGenerator:
    """Manages asynchronous batch generation in a separate thread."""
    
    def __init__(
        self,
        inference_client,
        num_batches_ahead: int = 1,
        max_queue_size: Optional[int] = None,
        generation_timeout: float = 600.0,
    ):
        """Initialize async batch generator.
        
        Args:
            inference_client: Client for inference server communication
            num_batches_ahead: Number of batches to generate ahead
            max_queue_size: Maximum number of batches in queue
            generation_timeout: Timeout for batch generation
        """
        self.inference_client = inference_client
        self.num_batches_ahead = num_batches_ahead
        self.generation_timeout = generation_timeout
        
        # Thread-safe queues
        self.request_queue = queue.Queue(maxsize=max_queue_size or 0)
        self.result_queue = queue.Queue()
        
        # Thread management
        self.stop_event = threading.Event()
        self.is_generating = False
        self._lock = threading.Lock()
        
        # Start generation thread
        self.generation_thread = threading.Thread(target=self._generation_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
    def _generation_worker(self):
        """Worker thread for batch generation."""
        print("[AsyncBatchGenerator] Generation worker thread started")
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.stop_event.is_set():
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=0.1)
                if request is None:  # Poison pill
                    print("[AsyncBatchGenerator] Received shutdown signal")
                    break
                    
                print(f"[AsyncBatchGenerator] Processing batch {request.batch_id}")
                # Generate batch
                with self._lock:
                    self.is_generating = True
                    
                start_time = time.time()
                try:
                    result = loop.run_until_complete(
                        self._generate_batch_async(request)
                    )
                    result.generation_time = time.time() - start_time
                    print(f"[AsyncBatchGenerator] Batch {request.batch_id} generated successfully in {result.generation_time:.2f}s")
                except Exception as e:
                    print(f"[AsyncBatchGenerator] ERROR generating batch {request.batch_id}: {e}")
                    result = BatchResult(
                        batch_id=request.batch_id,
                        processed_results={},
                        generation_time=time.time() - start_time,
                        error=str(e)
                    )
                    
                with self._lock:
                    self.is_generating = False
                    
                self.result_queue.put(result)
                print(f"[AsyncBatchGenerator] Batch {request.batch_id} added to result queue")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncBatchGenerator] Error in generation worker: {e}")
                
        print("[AsyncBatchGenerator] Generation worker thread shutting down")
        loop.close()
        
    async def _generate_batch_async(self, request: BatchRequest) -> BatchResult:
        """Generate batch asynchronously."""
        # Call inference client for generation
        try:
            # Remove timeout from env_inputs as it's not a parameter for generate_batch
            env_inputs = request.env_inputs.copy()
            if 'timeout' in env_inputs:
                del env_inputs['timeout']
                
            results = await self.inference_client.generate_batch(
                **env_inputs
            )
            
            return BatchResult(
                batch_id=request.batch_id,
                processed_results=results,
                generation_time=0.0  # Will be set by worker
            )
        except Exception as e:
            raise RuntimeError(f"Batch generation failed: {e}")
            
    def submit_batch(self, request: BatchRequest):
        """Submit batch for generation."""
        print(f"[AsyncBatchGenerator] Submitting batch {request.batch_id} for generation")
        self.request_queue.put(request)
        print(f"[AsyncBatchGenerator] Batch {request.batch_id} added to request queue (queue size: {self.request_queue.qsize()})")
        
    def get_batch(self, batch_id: int, timeout: Optional[float] = None) -> BatchResult:
        """Retrieve generated batch by ID."""
        print(f"[AsyncBatchGenerator] Waiting for batch {batch_id}")
        start_time = time.time()
        timeout = timeout or self.generation_timeout
        
        while True:
            try:
                # Check result queue
                result = self.result_queue.get(timeout=0.1)
                
                if result.batch_id == batch_id:
                    print(f"[AsyncBatchGenerator] Retrieved batch {batch_id} (waited {time.time() - start_time:.2f}s)")
                    return result
                else:
                    # Put back if not the right batch
                    print(f"[AsyncBatchGenerator] Got batch {result.batch_id} but waiting for {batch_id}, putting back")
                    self.result_queue.put(result)
                    
            except queue.Empty:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Batch {batch_id} generation timed out after {elapsed:.2f}s")
                if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10 seconds
                    print(f"[AsyncBatchGenerator] Still waiting for batch {batch_id} ({elapsed:.0f}s elapsed)")
                    
    def shutdown(self):
        """Shutdown the generation thread."""
        self.stop_event.set()
        self.request_queue.put(None)  # Poison pill
        self.generation_thread.join(timeout=5.0)


class AsyncDataLoaderWrapper:
    """Wrapper around dataloader that provides lookahead capabilities."""
    
    def __init__(self, dataloader, buffer_size: int = 10):
        """Initialize async dataloader wrapper.
        
        Args:
            dataloader: Base dataloader to wrap
            buffer_size: Size of lookahead buffer
        """
        self.dataloader = iter(dataloader)
        self.buffer_size = buffer_size
        self._buffer = []
        self._lock = threading.Lock()
        self._exhausted = False
        
        # Pre-fill buffer
        self._fill_buffer()
        
    def _fill_buffer_single(self):
        """Fill buffer with a single item."""
        try:
            item = next(self.dataloader)
            self._buffer.append(item)
        except StopIteration:
            self._exhausted = True
            
    def _fill_buffer(self):
        """Fill buffer to capacity."""
        with self._lock:
            while len(self._buffer) < self.buffer_size and not self._exhausted:
                self._fill_buffer_single()
                
    def __iter__(self):
        return self
        
    def __next__(self):
        """Get next batch."""
        with self._lock:
            if not self._buffer and self._exhausted:
                raise StopIteration
                
            # Get item from buffer
            if self._buffer:
                item = self._buffer.pop(0)
                
                # Refill buffer
                if not self._exhausted:
                    self._fill_buffer_single()
                    
                return item
            else:
                raise StopIteration
                
    def peek_ahead(self, n: int = 1) -> List[Any]:
        """Peek at the next n batches without consuming them.
        
        Args:
            n: Number of batches to peek ahead
            
        Returns:
            List of next n batches (may be shorter if not enough batches)
        """
        with self._lock:
            # Ensure buffer has enough items
            while len(self._buffer) < n and not self._exhausted:
                self._fill_buffer_single()
                
            return list(self._buffer)[:n]


class InferenceClient:
    """Base class for inference server client with weight synchronization."""
    
    def __init__(self, host: str, port: int):
        """Initialize inference client.
        
        Args:
            host: Inference server host
            port: Inference server port
        """
        self.host = host
        self.port = port
        self._weight_version = 0
        
    async def generate_batch(self, **kwargs) -> Dict[str, Any]:
        """Generate batch on inference server.
        
        This should be implemented by specific inference server clients.
        """
        raise NotImplementedError("Subclass must implement generate_batch")
        
    def update_weights(self, model_state: Dict[str, mx.array], adapter_config: Optional[Dict[str, Any]] = None):
        """Update model weights on inference server.
        
        Args:
            model_state: Model state dictionary
            adapter_config: Optional adapter configuration for LoRA
        """
        raise NotImplementedError("Subclass must implement update_weights")
        
    def get_weight_version(self) -> int:
        """Get current weight version."""
        return self._weight_version