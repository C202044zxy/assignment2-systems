import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        def hook(param: torch.Tensor):
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append((handle, param))

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad /= self.world_size
        self.handles.clear()