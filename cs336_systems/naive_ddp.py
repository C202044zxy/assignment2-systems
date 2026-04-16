import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from tests.common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)
from copy import deepcopy
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


def ddp_train(rank: int, world_size: int, model: nn.Module):
    _setup_process_group(rank, world_size, "gloo")
    dist.barrier()

    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = torch.randn(64, 5)

    # deepcopy both to avoid shared memory from mp.spawn
    reference_model = deepcopy(model)
    ddp_model = deepcopy(model)

    optimizer = optim.SGD(reference_model.parameters(), lr=0.1)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    shard_size = x.size(0) // world_size
    for iter in range(10):
        optimizer.zero_grad()
        ddp_optimizer.zero_grad()
        
        # train reference model on the full batch
        outputs = reference_model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        # train ddp model on sharded data
        shard_x = x[rank * shard_size : (rank + 1) * shard_size, :]
        shard_y = y[rank * shard_size : (rank + 1) * shard_size, :]
        shard_outputs = ddp_model(shard_x)
        shard_loss = loss_fn(shard_outputs, shard_y)
        shard_loss.backward()

        shard_grads = [p.grad for p in ddp_model.parameters() if p.grad is not None]
        flat = _flatten_dense_tensors(shard_grads)
        dist.all_reduce(flat)
        flat /= world_size
        reduced_grads = _unflatten_dense_tensors(flat, shard_grads)
        with torch.no_grad():
            for param, grad in zip(
                [p for p in ddp_model.parameters() if p.grad is not None],
                reduced_grads
            ):
                param.grad.copy_(grad)

        ddp_optimizer.step()

        for param, ddp_param in zip(reference_model.parameters(), ddp_model.parameters()):
            assert torch.allclose(param, ddp_param)
    _cleanup_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn=ddp_train, args=(world_size, ToyModel()), nprocs=world_size, join=True)