import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from typing import List
from collections import defaultdict

class Bucket:
    def __init__(self, params: List[torch.Tensor]):
        self.params = params
        self.world_size = dist.get_world_size()

        # assume all params have the same dtype
        total_elements = sum(p.numel() for p in params)
        self.buf = torch.zeros(total_elements, dtype=params[0].dtype, device=params[0].device)

        # allocate params to their slice
        self.params_to_slice = {}
        offset = 0
        for p in params:
            end = offset + p.numel()
            self.params_to_slice[id(p)] = slice(offset, end)
            offset = end
        
        self.handle = None
        self.pending = len(params)
    
    def mark_grad_ready(self, param: torch.Tensor):
        self.pending -= 1
        sl = self.params_to_slice[id(param)]
        self.buf[sl].copy_(param.grad.flatten())

        if self.pending == 0:
            self.handle = dist.all_reduce(self.buf, async_op=True)

    def finish(self):
        self.handle.wait()
        self.buf /= self.world_size
        for p in self.params:
            sl = self.params_to_slice[id(p)]
            p.grad.copy_(self.buf[sl].view_as(p.grad))
        
        self.handle = None
        self.pending = len(self.params)


class BucketDDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        self.buckets = self._build_buckets(bucket_size_mb)
        self.param_to_bucket = {
            id(p): bucket
            for bucket in self.buckets
            for p in bucket.params
        }
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _build_buckets(self, bucket_size_mb: float) -> List[Bucket]:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        # group by dtype, in reverse order
        dtype_to_params: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
        for p in reversed(list(self.module.parameters())):
            if p.requires_grad:
                dtype_to_params[p.dtype].append(p)
        
        # process each dtype independently
        buckets: List[Bucket] = []
        for _, params in dtype_to_params.items():
            current_bytes = 0
            current_params: List[torch.Tensor] = []

            for p in params:
                p_size = p.numel() * p.element_size()
                if p_size + current_bytes > bucket_size_bytes and current_params:
                    buckets.append(Bucket(current_params))
                    current_params = []
                    current_bytes = 0
                current_bytes += p_size
                current_params.append(p)
            
            if current_params:
                buckets.append(Bucket(current_params))
        return buckets

    def _make_hook(self):
        def hook(param: torch.Tensor):
            bucket = self.param_to_bucket[id(param)]
            bucket.mark_grad_ready(param)
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            bucket.finish()