"""5.5 连续批处理（评测来自仓库的 `continuous-batching` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass, field
from typing import List, Optional


class Request:
    """
    Represents a single inference request.
    
    Fields:
        request_id: Unique identifier
        input_ids: List of input token IDs (the prompt)
        max_gen_len: Maximum number of tokens to generate
        generated_ids: List of generated token IDs (grows over time)
        is_done: Whether generation is complete
        start_time: When the request started processing
        end_time: When the request finished
    """
    # TODO: Define the fields listed above
    # Use field(default_factory=list) for generated_ids
    # Use None as default for start_time and end_time
    # Use False as default for is_done
    ...
    
    def get_full_sequence(self):
        """Return the full token sequence (input + generated)."""
        # TODO
        ...
    
    def total_len(self):
        """Return the total sequence length."""
        # TODO
        ...


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler that dynamically adds/removes requests.
    
    - Maintains a waiting queue and an active batch.
    - At each step, fills empty slots, runs inference, evicts finished requests.
    """
    
    def __init__(self, max_batch_size, eos_token_id, vocab_size):
        # TODO: Initialize waiting_queue, active_batch, completed list,
        #       and store max_batch_size, eos_token_id, vocab_size
        ...
    
    def add_request(self, request):
        """Add a request to the waiting queue."""
        # TODO
        ...
    
    def _fill_batch(self):
        """Move requests from waiting queue to active batch (up to max_batch_size)."""
        # TODO: While there are empty slots and waiting requests,
        #       pop from waiting queue, set start_time, add to active batch
        ...
    
    def _evict_completed(self):
        """Remove completed requests from the active batch."""
        # TODO: Move done requests to completed list, set end_time,
        #       keep only active (not done) requests
        ...
    
    def step(self, model):
        """
        Run one generation step:
        1. Fill empty batch slots from waiting queue.
        2. For each active request, run the model and sample next token.
        3. Check for completion (EOS or max length).
        4. Evict completed requests.
        
        Returns:
            bool: True if there are still active or waiting requests.
        """
        # TODO: Implement the step logic described above.
        # For simplicity, process each active request individually
        # (no padding needed).
        ...
    
    def run(self, model):
        """
        Run the scheduler until all requests are completed.
        Returns the list of completed requests.
        """
        # TODO: Call step() in a loop until no more work to do
        ...
